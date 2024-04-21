import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

import math
import random

from PIL import Image, ImageOps, ImageEnhance
import numbers


# Explore the data(train data + test data)
train_df = pd.read_csv('../input/train.csv')   # 读取csv文件
n_train = len(train_df)  # 计算样本数量
n_pixels = len(train_df.columns) - 1  # 计算特征数量，-1是标签
n_class = len(set(train_df['label'])) # 计算标签数量，也就是有多少类，class
print('Number of training samples: {0}'.format(n_train))
print('Number of training pixels: {0}'.format(n_pixels))
print('Number of classes: {0}'.format(n_class))
test_df = pd.read_csv('../input/test.csv')
n_test = len(test_df)
n_pixels = len(test_df.columns)
print('Number of train samples: {0}'.format(n_test))
print('Number of test pixels: {0}'.format(n_pixels))


# Display the image
random_sel = np.random.randint(n_train, size=8) # 用Numpy库里的函数随机生成8个整数，构成一维数组，范围从0到n_train-1
# 构建包含8张灰度图像的图像网格
grid = make_grid(torch.Tensor((train_df.iloc[random_sel, 1:].as_matrix()/255.).reshape((-1, 28, 28))).unsqueeze(1), nrow=8)
# train_df.iloc[random_sel, 1:]：选择8个随机样本的特征部分
# as_matrix() 是将选定的数据转换为Numpy数组，即将8个特征部分转换为数组，形状为（8，784）
# reshape((-1, 28, 28))重新定形  -1在这里是一个占位符，表示根据其他维度的大小自动计算该维度的大小，在这里就变成（8，28，28）
# 说明是8个样本数，（28，28）是二维图像形状
# torch.Tensor 是将Numpy数组转化为tensor
# unsqueeze(1)是pytorch的函数，表示增加一个新维度，（1）表示在第二个维度上添加新的维度
# 最终形状：（8，1，28，28） 8个样本，1个通道（灰度图），28*28（高度*宽度/行*列）
plt.rcParams['figure.figsize'] = (16, 2) # 设置大小，宽度16英寸，高度为2英寸
plt.imshow(grid.numpy().transpose((1,2,0)))
# grid是tensor对象，然后转为numpy对象
# imshow规定的数据结构是（高度，宽度，通道数）——RGB   （高度，宽度）——灰度
# transpose（1，2，0）应该指transpose（1，2，0，3）？？
# 那这样（8，1，28，28）变成了（1，28，8，28） ？
# plt.imshow 函数将转置后的数组显示为一组灰度图像。
plt.axis('off')
print(*list(train_df.iloc[random_sel, 0].values), sep = ', ') # 打印标签


# Histogram of the classes
plt.rcParams['figure.figsize'] = (8, 5)
plt.bar(train_df['label'].value_counts().index, train_df['label'].value_counts())
plt.xticks(np.arange(n_class))
plt.xlabel('Class', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.grid('on', axis='y')




# Data loader
class MNIST_data(Dataset): # 继承Dataset类，类定义
    """MNIST data set"""

    def __init__(self, file_path,
                 transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,))])
                 ):
  # transform是预处理的组合：1. ToPILImage()就是将图像转换为PIL图像格式
                      #   2. ToTensor() Numpy转换为tensor
                      #   3. Normalize(mean=(0.5,), std=(0.5,))]就是归一化，将像素范围从【0，255】控制到【-1，1】
  # 通过以上的transform，先完成预处理
        df = pd.read_csv(file_path)  # 将数据加载到一个 DataFrame 中
        if len(df.columns) == n_pixels: # 31行所计算的图像的特征数
            # test data
            self.X = df.values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            # df.values 将 DataFrame 中的数据转换为一个 NumPy 数组
            # reshape将一维的数据转换为三维的（之前是784），现在为（样本数，28，28），每张图大小为28*28
            # .astype(np.uint8) 将图像数据转换为无符号8位整数类型，以节省内存空间
            # [:,:,:,None] 给图像数据添加一个额外的维度，将形状变为 (样本数, 28, 28, 1)
            # 这是因为 PyTorch 中要求图像数据有通道维度，而灰度图像只有一个通道，为1
            self.y = None # 标签
        else:
            # training data
            self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            # 同上面一样，但将第一列排除
            self.y = torch.from_numpy(df.iloc[:, 0].values)
              # df.iloc[:,0] 选择 DataFrame 的第一列，即标签列
              # torch.from_numpy(...) 将 NumPy 数组转换为 PyTorch 张量 tensor，得到训练数据的目标标签 y
        self.transform = transform

    def __len__(self): # 获取数据集的长度
        return len(self.X)  # 返回数据集的样本数量，即test data

    def __getitem__(self, idx):  # 根据索引获取单个样本
        if self.y is not None: # 判断数据集是否有标签，有说明是训练数据
            return self.transform(self.X[idx]), self.y[idx]
        # self.X[idx] 获取数据集中索引为 idx 的图像数据，然后通过 self.transform 进行预处理。
        # 再获取标签
        else: # 没有标签说明是测试数据
            return self.transform(self.X[idx])



        # Random Rotation transformation 旋转图像的操作
        def __init__(self, degrees, resample=False, expand=False, center=None):
            # 类函数，接受4个参数
            # degrees：旋转角度的范围
            # resample： 是否重新采样
            # expand： 是否扩展图像
            # center： 指定旋转中心
            if isinstance(degrees, numbers.Number):  # 判断degrees是否为单个数字
                if degrees < 0: # 角度必须大于0
                    raise ValueError("If degrees is a single number, it must be positive.")
                self.degrees = (-degrees, degrees)
                # 如果角度大于0，则变成一个范围，从 -degrees 到 degrees。
            else:  # 如果degrees不是单个数字，而是序列，则其长度必须为2
                if len(degrees) != 2:
                    raise ValueError("If degrees is a sequence, it must be of len 2.")
                self.degrees = degrees # 满足条件后，存储到self类

            self.resample = resample
            self.expand = expand
            self.center = center

        @staticmethod  # Python装饰器，表明下面的 get_params 方法是一个静态方法，可以直接通过类名调用。
        def get_params(degrees): # 静态方法，用于获取随机的degrees
            """Get parameters for ``rotate`` for a random rotation.
            Returns:
                sequence: params to be passed to ``rotate`` for random rotation.
            """
            angle = np.random.uniform(degrees[0], degrees[1])
            # 从指定的旋转角度范围中随机选择一个角度
            return angle

        def __call__(self, img): # 这是类的调用方法，用于对图像进行旋转操作，接受一个PIL图像为输入，返回旋转后的图像
            """
                img (PIL Image): Image to be rotated.
            Returns:
                PIL Image: Rotated image.
            """

            def rotate(img, angle, resample=False, expand=False, center=None):
                return img.rotate(angle, resample, expand, center)

                angle = self.get_params(self.degrees)  # 调用静态方法get_params所得到的degrees

                return rotate(img, angle, self.resample, self.expand, self.center)
                 # 根据rotate这个内部函数来实现图像的旋转




    # Random vertical and horizontal shift
    class RandomShift(object): # 首先定义一个类，继承了object类的所有功能
        def __init__(self, shift): # 平移
            self.shift = shift

        @staticmethod   # 仍旧是静态方法
        def get_params(shift): # 生成随机的平移参数，表示平移的最大范围
            """Get parameters for ``rotate`` for a random rotation.
            Returns:
                sequence: params to be passed to ``rotate`` for random rotation.
            """
            hshift, vshift = np.random.uniform(-shift, shift, size=2)
            # 生成2个随机数，分别表示水平和竖直方向的平移量
            return hshift, vshift

        def __call__(self, img): #这是类的调用方法。在这里，它被定义为对图像进行平移操作
            hshift, vshift = self.get_params(self.shift)

            return img.transform(img.size, Image.AFFINE, (1, 0, hshift, 0, 1, vshift), resample=Image.BICUBIC, fill=1)
            # 用img.transform对图像进行平移操作
            # (1, 0, hshift, 0, 1, vshift)实现平移，这个表示仿射变换矩阵
            # 通过指定 resample=Image.BICUBIC 来指定平移过程中的插值方式为双三次插值，保证图像的平滑过渡
            # fill=1 参数表示超出原图范围的区域用填充值 1 进行填充。



    # Load the data into Tensors
    batch_size = 64   # 每一次的输入数据的数量（样本数量）

    train_dataset = MNIST_data('../input/train.csv', transform=transforms.Compose(
        [transforms.ToPILImage(), RandomRotation(degrees=20), RandomShift(3),
         transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
    # 导入训练数据，之后进行预处理，旋转角度范围为[-20,20]，平移量为3，然后进行归一化
    test_dataset = MNIST_data('../input/test.csv')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True)
    # 创建了一个 PyTorch 的数据加载器 train_loader，用于批量加载训练数据集
    # 并且会指定输入的数据数量的大小以及是否在每次训练开始之前，需要随机打乱数据
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, shuffle=False) # 不需打乱数据



    # Visualize the transformation  显示图像
    rotate = RandomRotation(20) # 创建了一个 RandomRotation 类的实例 rotate，用于执行随机旋转操作，最大旋转角度为 20 度。
    shift = RandomShift(3)
    composed = transforms.Compose([RandomRotation(20),
                                   RandomShift(3)])
    # 创建了一个数据增强操作的组合，按照顺序先旋转再平移
    # Apply each of the above transforms on sample.
    fig = plt.figure() # 创建matplotlib窗口，用于展示结果图像
    sample = transforms.ToPILImage()(train_df.iloc[65, 1:].reshape((28, 28)).astype(np.uint8)[:, :, None])
    # 先找到第65个样本的像素数，再得到28*28 大小的图像数组
    # 增加一个新的维度  现在变为（28，28，1）
    for i, tsfrm in enumerate([rotate, shift, composed]): # 遍历三种操作
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1) # 在图形窗口中创建一个子图（行数，列数，索引）
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        ax.imshow((np.array(list(transformed_sample.getdata())), (-1, 28)), cmap='gray')
        # 灰度图显示
        # np.array转为Numpy数组
        # （-1，28）重塑形状，转换为二维的，28表示宽度
    plt.show()


    # 232-386行为核心
    # Network Structure
    class Net(nn.Module): # 定义一个新的类，继承nn.Moudle类
        # 在 PyTorch 中，所有的深度学习模型都需要继承自 nn.Module 类，并实现其 forward 方法来定义模型的前向传播过程。
        def __init__(self):
            super(Net, self).__init__()
            # super是内置函数，用于调用父类
            self.features = nn.Sequential(  # 卷积、归一化、激活函数层————特征提取
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                # 二维卷积层，输入通道数为1（灰度图），输出通道数为32，卷积核大小3*3，步长为1，填充为1
                nn.BatchNorm2d(32),
                # 归一化（批归一化）
                nn.ReLU(inplace=True),
                # 激活ReLU函数，引入非线性特性，通过 inplace=True 参数在原地执行激活函数，节省内存

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 二维最大池化层，降低空间尺寸，池化核大小为2*2，步长为2，即将特征图的高度和宽度都缩小一半

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.classifier = nn.Sequential( # 分类————全连接层
                nn.Dropout(p=0.5),
                #  Dropout 层，用于在训练过程中随机将输入元素置零，以减少过拟合。以0.5的概率置零
                nn.Linear(64 * 7 * 7, 512),
                # 全连接层，输入特征的维度是64*7*7，64是通道数，7*7是高度和宽度，64*7*7即特征图的总维度
                # 全连接层的作用就是将其平展为一个一维向量，维度是512
                nn.BatchNorm1d(512), # 对全连接层的数据归一化
                nn.ReLU(inplace=True), # 用激活函数ReLU

                nn.Dropout(p=0.5),
                nn.Linear(512, 512), # 新的全连接层，表示将前一层的输出映射到新的512维度的向量空间里
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),

                nn.Dropout(p=0.5),
                nn.Linear(512, 10), # 最后一个全连接层，将前一层的输出映射到维度为10的向量空间（在MNIST数据集中为0到9共10个类别）
            )


            for m in self.features.children(): # 遍历self.features的所有子层
                if isinstance(m, nn.Conv2d): # 如果是卷积子层
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    # m.kernel_size[0] * m.kernel_size[1] 表示卷积核的面积
                    # m.out_channels表示卷积核的个数
                    # 三个参数乘起来就表示卷积层的总参数数量
                    m.weight.data.normal_(0, math.sqrt(2. / n))  # Xavier初始化
                    # 正态分布的标准化，μ=0
                elif isinstance(m, nn.BatchNorm2d): # 如果是批归一层
                    m.weight.data.fill_(1)
                    # 这里的weight表示缩放因子，为1即表示不进行任何缩放
                    m.bias.data.zero_()
                    # 偏置因子

            for m in self.classifier.children():
                if isinstance(m, nn.Linear): # 如果是全连接层
                    nn.init.xavier_uniform(m.weight)
                    # 用Xavier方法对其权重初始化，pytorch现成的
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def forward(self, x): # 定义一个传播过程，即从输入到输出的数据流动过程
            x = self.features(x) # 首先通过特征提取部分得到特征图
            # x：是一个张量，其形状为 (batch_size, num_channels, height, width)
            x = x.view(x.size(0), -1) # 展平为一维向量
            # 现在变为了(batch_size, num_channels * height * width)，在全连接层之前，需要变成一维向量
            x = self.classifier(x) # 再用分类器进行处理（全连接）

            return x

        model = Net()
        # 创建模型 model：实例化 Net 类，得到一个包含特征提取和分类器的完整模型。
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        # 定义优化器 optimizer：使用 Adam 优化算法来优化模型参数，学习率为 0.003
        criterion = nn.CrossEntropyLoss()
        # 定义损失函数 criterion：使用交叉熵损失函数来衡量分类结果的准确性
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        # 定义学习率调度器 exp_lr_scheduler：使用 StepLR 调度器来动态调整学习率
        # 每经过 7 个 epoch，学习率会按照 gamma=0.1 的比例进行衰减。
        if torch.cuda.is_available(): # 检查是否有GPU
            model = model.cuda()
            criterion = criterion.cuda()



        # Traning and Evaluation
        def train(epoch): # 训练函数，epoch表示当前训练的轮数
            model.train()
            # 将模型设置为训练模式，这是为了确保在训练过程中启用Dropout和BatchNormalization等特定的操作
            exp_lr_scheduler.step()
            # 对优化器的学习率进行更新，这是为了在训练过程中进行学习率调整，通常使用学习率衰减策略

            for batch_idx, (data, target) in enumerate(train_loader):
                # 遍历dataloader，batch_idx为当前批次的索引，target是标签
                data, target = Variable(data), Variable(target)
                # 将输入的数据和标签组装到一起，构成pytorch的变量variable，后续可以自动求导

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad() # 清空之前保存的梯度信息，以便进行新一轮的梯度计算
                output = model(data) # 将输入数据 data 输入到模型中进行前向传播，得到输出 output。
                loss = criterion(output, target)
                # 计算模型的输出与真实标签之间的损失，criterion 是之前定义的交叉熵损失函数
                loss.backward() # 根据损失函数对模型的参数进行反向传播，计算参数的梯度
                optimizer.step() # 使用优化器更新模型的参数，即根据计算得到的梯度来更新参数

                if (batch_idx + 1) % 100 == 0: # 每经过100个批次则打印一次，监视
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                               100. * (batch_idx + 1) / len(train_loader), loss.data[0]))



        def evaluate(data_loader): # 评估训练好的数据
            model.eval() # 设置为评估模式，为了在评估过程中关闭Dropout和BatchNormalization等特定的操作，保持模型的固定状态。
            loss = 0
            correct = 0
            # 初始化损失和正确分类样本数的累积值，用于后续的计算

            for data, target in data_loader: # 遍历data_loader，获取所有的输入数据及其标签
                data, target = Variable(data, volatile=True), Variable(target)
                # 将输入数据和标签封装成 Variable 对象，并设置 volatile=True，表示在评估过程中不需要计算梯度，这样可以提高运算效率。
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                output = model(data)
                # 向前传播
                loss += F.cross_entropy(output, target, size_average=False).data[0]
                # 计算损失，F.cross_entropy 是PyTorch提供的交叉熵损失函数，用于计算模型输出与真实标签之间的损失
                pred = output.data.max(1, keepdim=True)[1]
                # 取得输出中概率最大的类别作为预测结果。
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # 计算预测正确的样本数
            loss /= len(data_loader.dataset)   # 平均损失

            print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
                loss, correct, len(data_loader.dataset),
                100. * correct / len(data_loader.dataset)))

            n_epochs = 1 # 进行训练和评估
            for epoch in range(n_epochs):
                train(epoch)
                evaluate(train_loader)



        # Prediction
        def prediciton(data_loader): # 预测
            model.eval()
            # 将模型设置为评估模式，这是为了在预测过程中关闭Dropout和BatchNormalization等特定的操作，保持模型的固定状态。
            test_pred = torch.LongTensor()
            # 用于存储预测结果，创建了一个空的LongTensor对象
            for i, data in enumerate(data_loader):
                data = Variable(data, volatile=True)
                if torch.cuda.is_available():
                    data = data.cuda()

                output = model(data)

                pred = output.cpu().data.max(1, keepdim=True)[1]
                test_pred = torch.cat((test_pred, pred), dim=0)

            return test_pred

        test_pred = prediciton(test_loader)
        out_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset) + 1)[:, None], test_pred.numpy()],
                              columns=['ImageId', 'Label'])
        out_df.head()