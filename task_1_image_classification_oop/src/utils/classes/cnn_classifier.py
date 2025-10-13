from .mnist_classifier_interface import MnistClassifierInterface
import torch
import torch.nn as nn
import torch.optim as optim

class CNNClassifier(MnistClassifierInterface):
    """
    Implementation of classification using a Convolutional Neural Network (PyTorch).
    Requires 4D input (N, C, H, W) where C=1 (grayscale).
    """
    def __init__(self, num_classes=10):
        # Define the network architecture
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # Input (N, 1, 28, 28)
                self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), # Output (N, 16, 28, 28)
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)) # Output (N, 16, 14, 14)
                self.layer2 = nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # Output (N, 32, 14, 14)
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)) # Output (N, 32, 7, 7)
                self.drop_out = nn.Dropout()
                self.fc = nn.Linear(7 * 7 * 32, num_classes)
            
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = x.reshape(x.size(0), -1) # Flatten (N, 7*7*32)
                x = self.drop_out(x)
                x = self.fc(x)
                return x

        # Define model, loss function, and optimizer
        self.model = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X_train, y_train, epochs=5, batch_size=64, **kwargs):
        """Trains the Convolutional Neural Network."""
        print(f"Training CNN for {epochs} epochs...")
        
        # Transform data to 4D Tensors that requires PyTorch CNN: (N, C, H, W) -> (N, 1, 28, 28)
        X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0 
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        # Create a DataLoader for batching
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        print("CNN training complete.")

    def predict(self, X_test):
        """Predicts labels using the trained CNN."""
        self.model.eval()
        # Transform data to 4D Tensors that requires PyTorch CNN: (N, C, H, W) -> (N, 1, 28, 28)
        X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.numpy()