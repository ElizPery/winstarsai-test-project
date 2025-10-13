from torchvision import datasets, transforms

def load_and_transform_mnist():
    """Loads MNIST data using torchvision and prepares numpy arrays."""
    # Define transformations: Convert to tensor, then normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load the training and testing datasets
    # Note: We load the full dataset here, and let the model classes handle splitting/preprocessing
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Convert PyTorch datasets to numpy arrays for compatibility
    X_train_full = train_dataset.data.numpy()
    y_train_full = train_dataset.targets.numpy()
    X_test_full = test_dataset.data.numpy()
    
    return X_train_full, y_train_full, X_test_full