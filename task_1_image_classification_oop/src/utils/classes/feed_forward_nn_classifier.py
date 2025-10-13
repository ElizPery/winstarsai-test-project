from .mnist_classifier_interface import MnistClassifierInterface
import torch
import torch.nn as nn
import torch.optim as optim

class FeedForwardNNClassifier(MnistClassifierInterface):
    """ Implementation of classification using a simple Feed-Forward Neural Network (PyTorch)."""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        # Define the network architecture
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, num_classes)
            
            def forward(self, x):
                x = x.view(-1, input_size) # Flatten the input
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        # Define model, loss function, and optimizer
        self.model = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, X_train, y_train, epochs=5, batch_size=64, **kwargs):
        """Trains the Feed-Forward Neural Network."""
        print(f"Training Feed-Forward Network for {epochs} epochs...")
        
        # Transform data to Tensors that requires PyTorch
        X_tensor = torch.tensor(X_train, dtype=torch.float32) / 255.0 # Normalize pixel values [0, 1]
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
        print("Feed-Forward Network training complete.")


    def predict(self, X_test):
        """Predicts labels using the trained Feed-Forward NN."""
        self.model.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32) / 255.0
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.numpy()
