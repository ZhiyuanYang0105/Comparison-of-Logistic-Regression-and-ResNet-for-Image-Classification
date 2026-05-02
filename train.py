from utils.dataset import get_data_loaders
from models.logistic import train_logistic
from models.resnet import get_resnet
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

train_loader, test_loader = get_data_loaders("data")

print("Train batches:", len(train_loader))
print("Test batches:", len(test_loader))


print("Training Logistic Regression...")
lr_model = train_logistic(train_loader)

X_test = []
y_test = []

for images, labels in test_loader:
    X_test.append(images.view(images.size(0), -1).numpy())
    y_test.append(labels.numpy())

X_test = np.vstack(X_test)
y_test = np.hstack(y_test)

print("LR Accuracy:", lr_model.score(X_test, y_test))

model = get_resnet()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("ResNet Accuracy:", correct / total)

lr_acc = lr_model.score(X_test, y_test)
resnet_acc = correct / total

plt.figure(figsize=(5,4))
plt.bar(["Logistic", "ResNet"], [lr_acc, resnet_acc], width=0.4)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.savefig("results/model_accuracy_comparison.png") 
plt.show()

# LR 预测
lr_pred = lr_model.predict(X_test)

cm = confusion_matrix(y_test, lr_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/lr_confusion_matrix.png") 
plt.show()

resnet_pred = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        resnet_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_test, resnet_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("ResNet Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/resnet_confusion_matrix.png") 
plt.show()