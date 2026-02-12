import torch
import torchvision
import torchvision.transforms as transforms
from model import CNN

def evaluate_model():
    print("Loading model and data")
    
    device = torch.device("cpu") 
    
    model = CNN()
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, 
                                           shuffle=False)

    confusion_matrix = torch.zeros(10, 10)

    print("Running predictions")
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for label, prediction in zip(labels, preds):
                confusion_matrix[label.long(), prediction.long()] += 1

    print("\n" + "="*55)
    print(f"{'Class':<10} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("="*55)

    f1_scores = []
    
    for i in range(10):
        tp = confusion_matrix[i, i]
        
        fp = confusion_matrix[:, i].sum() - tp
        
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp)        
        
        recall = tp / (tp + fn)                
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        f1_scores.append(f1.item())
        
        print(f"{i:<10} {precision:.4f}          {recall:.4f}          {f1:.4f}")


    total_correct = confusion_matrix.diag().sum()
    total_samples = confusion_matrix.sum()
    accuracy = total_correct / total_samples

    print("="*55)
    print(f"Overall Accuracy: {accuracy.item() * 100:.2f}%")
    print(f"Average F1-Score: {sum(f1_scores)/10:.4f}")
    print("="*55)

if __name__ == "__main__":
    evaluate_model()
