import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 加载数据集
train_dataset = datasets.ImageFolder(root='D:/model/resnet/new_test/new_test/pythonProject/train_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(root='D:/model/resnet/new_test/new_test/pythonProject/val_data', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. 加载预训练的 ResNet50 模型
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# 如果你想使用自己下载的 resnet50.pth
# model = models.resnet50(weights=None)
# model.load_state_dict(torch.load('resnet50.pth', weights_only=False))

# 4. 修改模型结构
num_ftrs = model.fc.in_features
# 假设你有 10 个类别
model.fc = nn.Linear(num_ftrs, 6)

# 5. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 6. 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    model.train()
    running_loss = 0.0
    train_loop = tqdm(train_loader, desc="Training", unit="batch")
    for inputs, labels in train_loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 更新进度条信息
        train_loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    print(f'Training Loss: {epoch_loss}')

    # 在验证集上评估模型
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    val_loop = tqdm(val_loader, desc="Validation", unit="batch")
    with torch.no_grad():
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条信息
            val_loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%')
