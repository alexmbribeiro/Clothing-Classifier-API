import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.datasets import fetch_openml
import numpy as np

# ----------------------------
# 1. Global Variables
# ----------------------------
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 5
SAMPLE_SIZE = 10000

# ----------------------------
# 2. Load & preprocess data
# ----------------------------
def load_data(sample_size=SAMPLE_SIZE):
    # Load Fashion-MNIST
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
    X = np.array(X) / 255.0
    y = np.array(y).astype(int)

    # Subsample
    X_small, y_small = resample(X, y, n_samples=sample_size, random_state=42, stratify=y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.2, random_state=42, stratify=y_small
    )

    # Reshape & resize for ResNet (224x224, 3 channels)
    X_train = X_train.reshape(-1, 28, 28)
    X_test  = X_test.reshape(-1, 28, 28)
    X_train = np.repeat(X_train[:, None, :, :], 3, axis=1)  # 3 channels
    X_test  = np.repeat(X_test[:, None, :, :], 3, axis=1)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    # Resize to 224x224 using transforms in DataLoader
    transform = transforms.Compose([
        transforms.Resize((224,224))
    ])

    # Wrap tensors in TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, test_loader

# ----------------------------
# 3. Model
# ----------------------------

def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model

# ----------------------------
# 4. Training & Evaluation
# ----------------------------
def train(model, train_loader, test_loader, epochs=EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    print(f"Test Accuracy: {100*correct/total:.2f}%")
    return model

# ----------------------------
# 5. Run
# ----------------------------
if __name__ == "__main__":
    train_loader, test_loader = load_data()
    model = get_model()
    trained_model = train(model, train_loader, test_loader)

    # Save the trained model
    torch.save(trained_model.state_dict(), "fashion_model.pt")
    print("Model saved to fashion_model.pt")
