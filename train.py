import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import CNN

def train_model():
    print("Setting up training")
    
    #Load Data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True)

    #Setup Model
    model = CNN()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Train Loop
    epochs = 10  
    print(f"Starting Training for {epochs} epochs")
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} finished. Avg Loss: {running_loss / len(trainloader):.4f}")

    #Save
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved to mnist_model.pth")

if __name__ == "__main__":
    train_model()