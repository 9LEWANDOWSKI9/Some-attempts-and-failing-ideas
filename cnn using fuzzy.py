import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pickle
import os
import torch.nn.functional as F
import requests
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
from sympy.tensor import tensor

torch.manual_seed(1)  # Set manual seed
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, Subset

if __name__ == '__main__':

    IMAGE_SIZE = 32

    mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

    # https://pytorch.org/vision/stable/transforms.html
    composed_train = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize the image in a 32X32 shape
         transforms.RandomRotation(20),  # Randomly rotate some images by 20 degrees
         transforms.RandomHorizontalFlip(0.1),  # Randomly horizontal flip the images
         transforms.ColorJitter(brightness=0.1,  # Randomly adjust color jitter of the images
                                contrast=0.1,
                                saturation=0.1),
         transforms.RandomAdjustSharpness(sharpness_factor=2,
                                          p=0.1),  # Randomly adjust sharpness
         transforms.ToTensor(),  # Converting image to tensor
         transforms.Normalize(mean, std),  # Normalizing with standard mean and standard deviation
         transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)])

    composed_test = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    # Load the data and transform the dataset
    train_dataset = dsets.CIFAR10(root='./datasets', train=True, download=True, transform=composed_train)
    validation_dataset = dsets.CIFAR10(root='./datasets', train=False, download=True, transform=composed_test)

    # Create train and validation batch for training
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=64)


    def show_data(img):
        try:
            plt.imshow(img[0])
        except Exception as e:
            print(e)
        print(img[0].shape, img[0].permute(1, 2, 0).shape)
        plt.imshow(img[0].permute(1, 2, 0))
        plt.title('y = ' + str(img[1]))
        plt.show()


    # We need to convert the images to numpy arrays as tensors are not compatible with matplotlib.
    def im_convert(tensor):
        # Lets
        img = tensor.cpu().clone().detach().numpy()  #
        img = img.transpose(1, 2, 0)
        img = img * np.array(tuple(mean)) + np.array(tuple(std))
        img = img.clip(0, 1)  # Clipping the size to print the images later
        return img


    # show_data(train_dataset[8])

    # Different classes in CIPHAR 10 dataset.
    classes = ('airplane',
               'automobile',
               'bird',
               'cat',
               'deer',
               'dog',
               'frog',
               'horse',
               'ship',
               'truck')

    # Define an iterable on the data
    data_iterable = iter(train_loader)  # converting our train_dataloader to iterable so that we can iter through it.
    images, labels = next(data_iterable)  # going from 1st batch of 100 images to the next batch
    fig = plt.figure(figsize=(20, 10))


    #
    # # Lets plot 50 images from our train_dataset
    # for idx in np.arange(10):
    #     ax = fig.add_subplot(2, 5, idx + 1, xticks=[], yticks=[])
    #
    #     # Note: imshow cant print tensor !
    #     # Lets convert tensor image to numpy using im_convert function for imshow to print the image
    #     plt.imshow(im_convert(images[idx]))
    #     ax.set_title(classes[labels[idx].item()])

    class CNN_generate_feature(nn.Module):
  # 隐藏层 num_repeats
  # 模糊逻辑 10个隐藏节点
        def __init__(self, out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0, num_repeats=10):
            super(CNN_generate_feature, self).__init__()
            self.cnn1 = nn.Conv2d(in_channels=3, out_channels=out_1, kernel_size=5, padding=2)
            self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
            self.conv_layers = nn.ModuleList(
                [nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=5, padding=2) for _ in
                 range(num_repeats)])


            self.num_repeats = num_repeats

 # 10*64*128*32*32 卷积输出的特征向量
            self.random_d_matrix = nn.Parameter((torch.ones(10, 64, 1, 32, 32) ).float() )
                         # 64个1*32*32

            self.random_c_matrix = nn.Parameter((torch.ones(10, 64, 128, 32, 32) ).float() )
 # 通过不断更新d矩阵来决定哪些特征要哪些特征不要
#  64*32*32*3     d 64*32*32
            self.fc1 = nn.Linear(1280, 1000)
            self.drop = nn.Dropout(p=p)
            self.fc1_bn = nn.BatchNorm1d(1000)
            self.fc2 = nn.Linear(1000, 1000)
            self.fc2_bn = nn.BatchNorm1d(1000)
            self.fc3 = nn.Linear(1000, 1000)
            self.fc3_bn = nn.BatchNorm1d(1000)
            self.fc4 = nn.Linear(1000, 1000)
            self.fc4_bn = nn.BatchNorm1d(1000)
            self.fc5 = nn.Linear(1000, 10)
            self.fc5_bn = nn.BatchNorm1d(10)






        def forward(self, x):
            x = self.cnn1(x)
            x = torch.relu(x)

            x = self.cnn2(x)
            x = torch.relu(x)

            x = [conv(x) for conv in self.conv_layers]
            x = torch.stack(x)

            # 使用 sigmoid 函数将参数限制在 0 到 1 之间

            random_c_matrix = self.random_c_matrix


            # 空间注意力机制
            # 思路，目的，特征 还是性能

            # 模型新 参考意义 有什么用



            random_d_matrix = self.random_d_matrix




            x = x * random_c_matrix

            x = x * random_d_matrix

            new_shape = (10, 64, 128, -1)
            x = x.view(*new_shape)
            x = 1 - torch.abs(x)
            x = torch.prod(x, dim=-1)
            x = x.view(x.size(1), -1)

            x = self.fc1(x)
            x = self.fc1_bn(x)
            x = F.relu(self.drop(x))
            x = self.fc2(x)
            x = self.fc2_bn(x)
            x = F.relu(self.drop(x))
            x = self.fc3(x)
            x = self.fc3_bn(x)
            x = F.relu(self.drop(x))
            x = self.fc4(x)
            x = self.fc4_bn(x)
            x = F.relu(self.drop(x))
            x = self.fc5(x)
            x = self.fc5_bn(x)

            return x
    #########################  测试

    model = CNN_generate_feature(out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0.5, num_repeats=10)


    # 假设 input_tensor 是一个大小为 (batch_size, channels, height, width) 的输入张量
    input_tensor = torch.randn(64, 3, 32, 32)

    # 进行前向传播
    outputs = model(input_tensor)

    print(outputs.shape)

    # 创建模型实例

    model = CNN_generate_feature(out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0.5, num_repeats=10)

    # 假设 input_tensor 是一个大小为 (batch_size, channels, height, width) 的输入张量
    input_tensor = torch.randn(64, 3, 32, 32)

    # 进行前向传播
    outputs = model(input_tensor)

    # print(outputs)  # torch.Size([10, 64, 128, 32, 32]) 有10 个并列隐藏层 有64个batch 有 128 个特征 有 32* 32 个维度

    import torch.optim as optim

    model = CNN_generate_feature(out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0.5)
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 10

    accuracy_list = []
    train_cost_list = []
    val_cost_list = []

    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:  # 假设train_dataloader是你的训练数据迭代器
            optimizer.zero_grad()


            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        train_accuracy = correct_predictions / total_samples
        accuracy_list.append(train_accuracy)
        train_cost_list.append(running_loss / len(train_loader))

        # 在验证集上进行评估
        model.eval()  # 将模型设置为评估模式
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_inputs, val_labels in validation_loader:  # 假设val_dataloader是你的验证数据迭代器
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()

                _, val_predicted = torch.max(val_outputs, 1)
                val_correct += (val_predicted == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_accuracy = val_correct / val_total
        val_cost_list.append(val_loss / len(validation_loader))

        # 打印每个epoch的信息
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_cost_list[-1]}, Train Accuracy: {accuracy_list[-1]}, Val Loss: {val_cost_list[-1]}, Val Accuracy: {val_accuracy}")

        trained_random_d_matrix = model.random_d_matrix.data

        trained_random_c_matrix = model.random_c_matrix.data

        matrices_c_list = tensor.chunk(trained_random_c_matrix.size(0), dim=0)

        matrices_d_list = tensor.chunk(trained_random_d_matrix.size(0), dim=0)


        # 将元组转换为列表

        matrices_c_list = list(matrices_c_list)




        matrices_d_list = list(matrices_d_list)










    # def train_model(model, train_loader, validation_loader, optimizer, n_epochs=20):
    #
    #     # Global variable
    #     N_test = len(validation_dataset)
    #     accuracy_list = []
    #     train_cost_list = []
    #     val_cost_list = []
    #
    #     for epoch in range(n_epochs):
    #         train_COST = 0
    #         for x, y in train_loader:
    #             model.train()
    #             optimizer.zero_grad()
    #             z = model(x)
    #
    #
    #             loss = criterion(z, y)
    #             loss.backward()
    #             optimizer.step()
    #             train_COST += loss.item()
    #
    #         train_COST = train_COST / len(train_loader)
    #         train_cost_list.append(train_COST)
    #         correct = 0
    #
    #         # Perform the prediction on the validation data
    #         val_COST = 0
    #         for x_test, y_test in validation_loader:
    #             model.eval()
    #             z = model(x_test)
    #             val_loss = criterion(z, y_test)
    #             _, yhat = torch.max(z.data, 1)
    #             correct += (yhat == y_test).sum().item()
    #             val_COST += val_loss.item()
    #
    #         val_COST = val_COST / len(validation_loader)
    #         val_cost_list.append(val_COST)
    #
    #         accuracy = correct / N_test
    #         accuracy_list.append(accuracy)
    #
    #         print("--> Epoch Number : {}".format(epoch + 1),
    #               " | Training Loss : {}".format(round(train_COST, 4)),
    #               " | Validation Loss : {}".format(round(val_COST, 4)),
    #               " | Validation Accuracy : {}%".format(round(accuracy * 100, 2)))
    #
    #     return accuracy_list, train_cost_list, val_cost_list
    #

    # ...

    # 创建模型实例
    # model = CNN_generate_feature(out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0.5)
    #
    # # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    # learning_rate = 0.1
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.2)
    #
    # # 调用 train_model 函数进行模型训练
    # accuracy, train_cost, val_cost = train_model(model=model, n_epochs=200,
    #                                              train_loader=train_loader,
    #                                              validation_loader=validation_loader,
    #                                              optimizer=optimizer)
