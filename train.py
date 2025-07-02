import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from datetime import datetime

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc1(x)
        return x

def run_training(dataset_name='CIFAR10', batch_size=64, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_shape = (3, 32, 32)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("CNN_Voila_Training")

    with mlflow.start_run():
        # Hiperparametreleri logla
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        # Model açıklamasını ve örnek input-output signature'ı hazırla
        model_description = str(model)
        example_input = torch.rand(1, 3, 32, 32)  # CIFAR10 boyutları
        example_output = model(example_input.to(device)).cpu().detach()
        signature = infer_signature(example_input.cpu().numpy(), example_output.numpy())

        # Eğitim döngüsü
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f}")

        # Test doğruluğu
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Accuracy: {acc:.2f}%")
        mlflow.log_metric("accuracy", acc)

        # Modeli versiyonla ve signature ile birlikte kaydet
        mlflow.set_tag("model_description", model_description)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            input_example=example_input.cpu(),
            signature=signature,
            registered_model_name="CNN_Model",
            name="CNN_Model"
        )

        print("Model ve sonuçlar MLflow'a kaydedildi.")