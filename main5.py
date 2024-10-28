import torch
from assets.models.lenet5 import QuantizedLeNet5
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU
import copy
from torchvision import datasets, transforms
from utils.evaluate import evaluate
from utils.loader import load_mnist_data, load_mnist_test_data
from torch.utils.data import DataLoader
from tools.neuron_pruning import verify_pruned_model
from tools.structural_prune import structural_pruning
import matplotlib.pyplot as plt

def count_parameters(model):
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_structure(model):
    """Print detailed model structure including quantization."""
    for name, module in model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            print(f"\nLayer: {name}")
            print(f"Type: {type(module).__name__}")
            print(f"Input shape: {module.in_channels if isinstance(module, QuantConv2d) else module.in_features}")
            print(f"Output shape: {module.out_channels if isinstance(module, QuantConv2d) else module.out_features}")
            print(f"Weight shape: {module.weight.shape}")
            if module.bias is not None:
                print(f"Bias shape: {module.bias.shape}")
            print(f"Weight quant: {type(module.weight_quant).__name__}")
            try:
                bit_width = module.weight_quant.bit_width if hasattr(module.weight_quant, 'bit_width') else "N/A"
                print(f"Bit width: {bit_width}")
            except Exception as e:
                print(f"Bit width: Unable to determine ({str(e)})")
        elif isinstance(module, (QuantReLU, torch.nn.AvgPool2d)):
            print(f"\nLayer: {name}")
            print(f"Type: {type(module).__name__}")

def compute_sensitivity_scores(model, mask, device):
    """Calculate sensitivity scores based on weight magnitudes for each layer."""
    sensitivity_scores = {}
    model = model.to(device)
    for name, module in model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear)) and name in mask:
            weights = module.weight.data.abs()  # Calculate absolute weight values
            scores = weights.mean(dim=0) if isinstance(module, QuantLinear) else weights.mean(dim=[1, 2, 3])
            sensitivity_scores[name] = scores.to(device)  # Store mean score per neuron/channel
    return sensitivity_scores

def plot_training_progress(epochs, accuracy, loss):
    """Plot training accuracy and loss progression."""
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(range(1, epochs + 1), accuracy, color='tab:blue', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    ax2.plot(range(1, epochs + 1), loss, color='tab:red', label='Loss')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title("Training Accuracy and Loss Over Epochs")
    
    # Save the plot to the assets directory
    plt.savefig("assets/training_progress.png")
    plt.close(fig)  # Close the figure to free up memory

def main():
    try:
        # Load the soft-pruned model
        original_model = QuantizedLeNet5()
        checkpoint = torch.load('assets/lenet5_quantized_pruned_new.pth')
        original_model.load_state_dict(checkpoint['model_state_dict'])
        mask = checkpoint['mask']

        # Load the datasets and DataLoader for training
        train_loader = load_mnist_data()
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

        print("\nOriginal (soft-pruned) model structure:")
        print_model_structure(original_model)

        # Verify and analyze sensitivity of the original model
        print("\nVerifying and analyzing sensitivity of the original model:")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sensitivity_scores = compute_sensitivity_scores(original_model, mask, device)
        verify_pruned_model(original_model, mask)

        # Structurally prune based on sensitivity analysis
        print("\nCreating structurally pruned model with sensitivity-based pruning...")
        pruned_model = structural_pruning(original_model, mask, train_loader, sensitivity_scores)

        # Initialize training parameters for visualization
        epochs = 3  # Change if needed
        train_accuracy = []
        train_loss = []

        # Training loop with accuracy and loss tracking
        pruned_model = pruned_model.to(device)
        optimizer = torch.optim.Adam(pruned_model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            pruned_model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = pruned_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total
            train_accuracy.append(accuracy)
            train_loss.append(epoch_loss / len(train_loader))

            print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Plot the training progress
        plot_training_progress(epochs, train_accuracy, train_loss)

        # Compare models
        original_accuracy = evaluate(original_model, test_dataset)
        original_params = count_parameters(original_model)
        
        pruned_accuracy = evaluate(pruned_model, test_dataset)
        pruned_params = count_parameters(pruned_model)
        
        param_reduction = (original_params - pruned_params) / original_params * 100

        print(f"\nResults:")
        print(f"Original model - Accuracy: {original_accuracy:.4f}, Parameters: {original_params}")
        print(f"Structural Pruned model - Accuracy: {pruned_accuracy:.4f}, Parameters: {pruned_params}")
        print(f"Parameter reduction: {param_reduction:.2f}%")

        # Save pruned model if performance drop is within 5%
        if abs(original_accuracy - pruned_accuracy) < 0.05:
            torch.save({
                'model_state_dict': pruned_model.state_dict(),
                'mask': mask,
                'accuracy': pruned_accuracy,
                'original_accuracy': original_accuracy,
                'params_reduced': param_reduction
            }, "assets/lenet5_quantized_structurally_pruned_sensitive.pth")
            print("\nStructurally pruned model with sensitivity pruning saved successfully.")
        else:
            print("\nWarning: Large accuracy drop detected. Model not saved.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()