import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import os

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dir = "/Users/busragural/Desktop/4.1/Deep Learning/Proje/test"
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classes = test_dataset.classes
print("Classes:", classes)

model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, len(classes))
)

pth_file = "./resnet50_61acc.pth"

model.load_state_dict(torch.load(pth_file, map_location=device))
model = model.to(device)
model.eval()

def test_model(model, test_loader, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(12, 8))
    sn.heatmap(df_cm, annot=True, cmap="OrRd", fmt="d", annot_kws={"size": 10}, cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=classes))


test_model(model, test_loader, classes)
