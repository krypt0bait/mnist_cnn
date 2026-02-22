import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
from model import CNN

def train_model():
    print("Setting up training")
    
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    
    valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1000, shuffle=False)

   
    model = CNN()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    train_losses = []
    val_losses = []


    epochs = 10  
    print(f"Starting Training for {epochs} epochs")
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in valloader:
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_labels)
                running_val_loss += v_loss.item()
        
        avg_val_loss = running_val_loss / len(valloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved to mnist_model.pth")

    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig('loss_graph.png') 
    plt.show()

if __name__ == "__main__":
    train_model()