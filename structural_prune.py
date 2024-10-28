import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU
from brevitas.quant import (
    Int8ActPerTensorFloat, 
    Uint8ActPerTensorFloat, 
    Int8WeightPerTensorFloat, 
    Int8WeightPerChannelFloat
)
from brevitas.core.scaling import StatsFromParameterScaling
from brevitas.core.restrict_val import RestrictValueType
from assets.models.lenet5 import QuantizedLeNet5
from utils.evaluate import evaluate

# Quantization Classes
class Common8bitWeightPerTensorQuant(Int8WeightPerTensorFloat):
    scaling_min_val = 2e-16

class Common8bitWeightPerChannelQuant(Int8WeightPerChannelFloat):
    scaling_per_output_channel = True

class Common4bitWeightPerTensorQuant(Common8bitWeightPerTensorQuant):
    bit_width = 4

class Common4bitWeightPerChannelQuant(Common8bitWeightPerChannelQuant):
    bit_width = 4

class Common8bitActQuant(Int8ActPerTensorFloat):
    scaling_min_val = 2e-16
    restrict_scaling_type = RestrictValueType.LOG_FP

class Common4bitActQuant(Common8bitActQuant):
    bit_width = 4

# Network Structure Analysis
def analyze_network_structure(original_model, mask):
    """Maps network structure and dependencies."""
    network_structure, layer_info = [], {}
    prev_learnable = None

    for name, module in original_model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            # For fc3, ensure all neurons are active
            if name == 'fc3':
                active_indices = torch.arange(module.out_features)
            else:
                active_indices = torch.nonzero(mask[name]).squeeze(1)
            layer_info[name] = {
                'type': type(module),
                'active_indices': active_indices,
                'active_channels': len(active_indices),
                'prev_learnable': prev_learnable,
                'dependencies': []
            }
            print(f"{name} active indices: {layer_info[name]['active_indices']}")  # Debugging line
            prev_learnable = name
            network_structure.append(name)
        elif isinstance(module, (QuantReLU, torch.nn.AvgPool2d)):
            if prev_learnable:
                layer_info[prev_learnable]['dependencies'].append((name, type(module)))
            network_structure.append(name)

    return network_structure, layer_info

def create_conv_layer(original_module, active_indices, prev_active_indices):
    """Creates a new Conv layer with pruned channels."""
    is_8bit = isinstance(original_module.weight_quant, (Common8bitWeightPerChannelQuant, Common8bitWeightPerTensorQuant))

    # Ensure active_indices and prev_active_indices are tensors
    active_indices = torch.tensor(active_indices) if isinstance(active_indices, list) else active_indices
    prev_active_indices = torch.tensor(prev_active_indices) if isinstance(prev_active_indices, list) else prev_active_indices

    print(f"Creating conv layer with {len(active_indices)} output channels and {len(prev_active_indices)} input channels.")  # Debugging line

    # Create a new convolution layer with pruned input and output channels
    new_conv = QuantConv2d(
        in_channels=len(prev_active_indices),
        out_channels=len(active_indices),
        kernel_size=original_module.kernel_size,
        stride=original_module.stride,
        padding=original_module.padding,
        bias=original_module.bias is not None,
        weight_quant=Common8bitWeightPerChannelQuant if is_8bit else Common4bitWeightPerChannelQuant,
        input_quant=Common8bitActQuant if is_8bit else Common4bitActQuant,
        output_quant=Common8bitActQuant if is_8bit else Common4bitActQuant,
        return_quant_tensor=True
    )

    with torch.no_grad():
        weight_data = original_module.weight.data

        # Slice weights based on active output and input channels
        weight_data = weight_data[active_indices][:, prev_active_indices]
        print(f"Weight data shape after slicing: {weight_data.shape}")  # Debugging line

        new_conv.weight.data.copy_(weight_data)

        if original_module.bias is not None:
            new_conv.bias.data.copy_(original_module.bias.data[active_indices])

    return new_conv


def create_linear_layer(original_module, active_indices, prev_active_indices, in_features, is_first_linear=False):
    """Creates a new Linear layer with pruned neurons and calculated input shape."""
    active_indices = torch.tensor(active_indices) if isinstance(active_indices, list) else active_indices
    prev_active_indices = torch.tensor(prev_active_indices) if isinstance(prev_active_indices, list) else prev_active_indices

    # Choose the correct quantization classes based on the original layer's quantization
    is_8bit = isinstance(original_module.weight_quant, (Common8bitWeightPerChannelQuant, Common8bitWeightPerTensorQuant))

    new_linear = QuantLinear(
        in_features=in_features,
        out_features=len(active_indices),
        bias=original_module.bias is not None,
        weight_quant=Common8bitWeightPerChannelQuant if is_8bit else Common4bitWeightPerChannelQuant,
        input_quant=Common8bitActQuant if is_8bit else Common4bitActQuant,
        output_quant=Common8bitActQuant if is_8bit else Common4bitActQuant,
        return_quant_tensor=True
    )

    with torch.no_grad():
        if is_first_linear:
            # Compute input feature indices based on active channels
            feature_map_size = 5 * 5  # After pooling
            input_feature_indices = []
            for c in prev_active_indices:
                start_idx = c.item() * feature_map_size
                end_idx = (c.item() + 1) * feature_map_size
                input_feature_indices.extend(range(start_idx, end_idx))
            input_feature_indices = torch.tensor(input_feature_indices)
        else:
            # For subsequent linear layers, prev_active_indices correspond to input features
            input_feature_indices = prev_active_indices

        original_weights = original_module.weight.data[active_indices][:, input_feature_indices]
        new_linear.weight.data.copy_(original_weights)

        if original_module.bias is not None:
            new_linear.bias.data.copy_(original_module.bias.data[active_indices])

    return new_linear


def compute_linear_in_features(prev_active_indices, original_model):
    # Create a dummy input tensor with compatible dimensions
    dummy_input = torch.randn(1, 1, 28, 28)  # MNIST image size
    for name, module in original_model.named_children():
        if isinstance(module, QuantConv2d):
            # Only apply indexing if there are multiple channels
            if dummy_input.size(1) > 1:
                dummy_input = torch.index_select(dummy_input, 1, prev_active_indices)
            dummy_input = module(dummy_input)
            prev_active_indices = torch.arange(module.out_channels)  # Update for the next layer's input channels
        elif isinstance(module, torch.nn.AvgPool2d):
            dummy_input = module(dummy_input)
        else:
            continue

    # Flatten the output to simulate transition to a fully connected layer
    dummy_input = dummy_input.tensor if hasattr(dummy_input, 'tensor') else dummy_input
    return dummy_input.view(1, -1).size(1)




def verify_model(new_model, original_model, layer_info):
    """
    Verifies that the pruned model has the correct structure by comparing key attributes
    between the pruned model and the original model's pruned structure.
    """
    prev_out_channels = 1  # Starting input channels for MNIST data

    for name in layer_info:
        layer_details = layer_info[name]
        original_module = getattr(original_model, name, None)
        new_module = getattr(new_model, name, None)

        if original_module is None or new_module is None:
            print(f"Layer {name} missing in one of the models.")
            return False

        # Check if types match
        if type(original_module) != type(new_module):
            print(f"Layer type mismatch in {name}: {type(original_module)} vs {type(new_module)}")
            return False

        # Expected output channels based on updated `active_channels`
        if isinstance(original_module, QuantConv2d):
            pruned_out_channels = int(layer_details['active_channels'])
            pruned_in_channels = int(prev_out_channels)  # Input channels from previous layer’s pruned channels
            expected_weight_shape = (pruned_out_channels, pruned_in_channels, *original_module.weight.shape[2:])

            # Check weight shape
            if new_module.weight.shape != expected_weight_shape:
                print(f"Weight shape mismatch in {name}: expected {expected_weight_shape}, got {new_module.weight.shape}")
                return False

            prev_out_channels = pruned_out_channels  # Update for the next layer

        elif isinstance(original_module, QuantLinear):
            expected_out_features = int(layer_details['active_channels'])

            # For the first linear layer, calculate `expected_in_features`
            if name == 'fc1':
                feature_map_size = 5 * 5  # After pooling
                expected_in_features = prev_out_channels * feature_map_size
            else:
                expected_in_features = int(prev_out_channels)  # From previous layer

            # Check linear layer shapes
            if new_module.weight.shape != (expected_out_features, expected_in_features):
                print(f"Weight shape mismatch in {name}: expected ({expected_out_features}, {expected_in_features}), got {new_module.weight.shape}")
                return False

            prev_out_channels = expected_out_features  # Update for the next layer

    print("Model verification passed!")
    return True


# Main Structural Pruning Function
def structural_pruning(original_model, mask, train_loader, sensitivity_scores=None):
    """Prune the model structure based on sensitivity scores and fine-tune it."""

    # Initialize new model and analyze network structure
    new_model = QuantizedLeNet5()
    network_structure, layer_info = analyze_network_structure(original_model, mask)
    new_model.quant_inp = copy.deepcopy(original_model.quant_inp)
    
    prev_active_indices = None  # To track the previous layer's active indices
    
    # Iterating through each layer for pruning and reassigning
    for name in network_structure:
        module = getattr(original_model, name)
        print(f"Processing layer {name}...")

        if isinstance(module, (QuantConv2d, QuantLinear)):
            active_indices = layer_info[name]['active_indices']

            if sensitivity_scores and name in sensitivity_scores and name != 'fc3':
                # Filter active indices based on sensitivity score
                threshold = sensitivity_scores[name].mean() * 0.5
                active_indices = [i for i in active_indices if sensitivity_scores[name][i] >= threshold]
                layer_info[name]['active_indices'] = active_indices
                layer_info[name]['active_channels'] = len(active_indices)
                print(f"{name} pruned to {len(active_indices)} active indices")

            if isinstance(module, QuantConv2d):
                if prev_active_indices is None:
                    prev_active_indices = torch.arange(module.in_channels)
                new_layer = create_conv_layer(module, active_indices, prev_active_indices)
                prev_active_indices = active_indices  # Update previous active indices for next layer
                setattr(new_model, name, new_layer)

            elif isinstance(module, QuantLinear):
                in_features = len(prev_active_indices) * (5 * 5 if name == 'fc1' else 1)
                if name == 'fc3':
                    active_indices = layer_info[name]['active_indices']
                    print(f"Ensuring {name} retains all output neurons: {len(active_indices)}")
                new_layer = create_linear_layer(module, active_indices, prev_active_indices, in_features, name == 'fc1')
                prev_active_indices = active_indices
                setattr(new_model, name, new_layer)

            # Copy dependencies (e.g., quantization and other attributes)
            for dep_name, _ in layer_info[name]['dependencies']:
                setattr(new_model, dep_name, copy.deepcopy(getattr(original_model, dep_name)))

        else:
            # Directly copy non-pruned layers (e.g., ReLU, Pooling)
            setattr(new_model, name, copy.deepcopy(module))

    # Verify and fine-tune the pruned model
    if not verify_model(new_model, original_model, layer_info):
        raise RuntimeError("Model verification failed!")
    
    # Fine-tuning the pruned model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_model = new_model.to(device)
    optimizer = optim.Adam(new_model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
    best_accuracy = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(3):
        new_model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = new_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        accuracy = 100 * correct / total
        scheduler.step(accuracy)

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
        # Save best state
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(new_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        new_model.load_state_dict(best_state)
        print(f"Best model restored with accuracy: {best_accuracy:.4f}")

    return new_model