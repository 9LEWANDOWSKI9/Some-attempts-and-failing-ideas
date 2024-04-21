import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from clustering import AFS

from collections import Counter
from operator import itemgetter




class CustomModel(nn.Module):
    def __init__(self, input_size, afs_output_size):
        super(CustomModel, self).__init__()
        self.conv_layer = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pooling_layer = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.afs_output_size = afs_output_size

        # 添加其它层...

        self.dense_layer = nn.Linear(32 + afs_output_size, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x, afs_output):
        x = self.conv_layer(x)
        x = self.pooling_layer(x)
        x = self.flatten(x)

        # 添加其它层的forward...

        x = torch.cat((x, afs_output), dim=1)
        x = self.dense_layer(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


# 数据准备
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AFS 聚类
afs = AFS()
parameters = afs.get_parameters(X_train)
weight_matrix = afs.generate_weight_matrix(X_train, parameters)
descriptions_train = afs.generate_descriptions(X_train, weight_matrix, neighbour=5, feature=3, epsilon=0.1)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
descriptions_train_tensor = torch.tensor(descriptions_train, dtype=torch.float32)

# 构建 DataLoader
train_dataset = TensorDataset(X_train_tensor.unsqueeze(1), descriptions_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建模型
afs_output_size = descriptions_train.shape[1]
model = CustomModel(X_train.shape[1], afs_output_size)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, afs_output, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs, afs_output)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

# 评估模型
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
descriptions_test = afs.generate_descriptions(X_test, weight_matrix, neighbour=5, feature=3, epsilon=0.1)
descriptions_test_tensor = torch.tensor(descriptions_test, dtype=torch.float32)

with torch.no_grad():
    predictions = model(X_test_tensor.unsqueeze(1), descriptions_test_tensor)
    predictions = (predictions > 0.5).float()
    accuracy = torch.sum(predictions == y_test) / float(len(y_test))

print(f"Test Accuracy: {accuracy.item()}")
