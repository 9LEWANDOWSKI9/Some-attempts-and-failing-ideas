#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
##  Module run.py
##  Python3 based module
##  Created:  Mon 10 17:21:55 GMT 2020 by Samaresh Nayak

"""
This module follows a fuzzy deep neural network model for image classification
on the CIFAR-10 dataset.

optional arguments:
  -h, --help            show this help message and exit
  --learning-rate LEARNING_RATE
                        Learning Rate of your classifier. Default 0.0001
  --epoch EPOCHS        Number of times you want to train your data. Default
                        100
  --batch-size BATCH_SIZE
                        Batch size for prediction. Default=16.
  --colour-image        Passing this argument will keep the coloured image
                        (RGB) during training. Default=False.
  --membership-layer-units MEMBERSHIP_LAYER_UNITS
                        Defines the number of units/nodes in the Membership
                        Function Layer
  --first-dr-layer-units DR_LAYER_1_UNITS
                        Defines the number of units in the first DR Layer
  --second-dr-layer-units DR_LAYER_2_UNITS
                        Defines the number of units in the second DR Layer
  --fusion-dr-layer-units FUSION_DR_LAYER_UNITS
                        Defines the number of units in the Fusion DR Layer
  --fusion-dr-layer-units FUSION_DR_LAYER_UNITS
                        Defines the number of units in the Fusion DR Layer
  --hide-graph          Hides the graph of results displayed via matplotlib

example usage:
    run.py --epoch 100 --batch-size 8 --learning-rate 0.001
           --membership-layer-units 256 --first-dr-layer-units 128
           --second-dr-layer-units 64
"""

__author__ ='Samaresh Nayak'
__version__ = '1.0'

import argparse  # 解析命令行参数的库
from pprint import pprint  # 美化打印结构

import numpy as np  # 处理数组和数值计算
from keras.datasets import cifar10   # 导入cifar10，包含对于该数据集的加载功能
from keras.utils import to_categorical  # 独热编码
from matplotlib import pyplot   # 可视化数据
from tensorflow.keras import Model  # 定义模型
from tensorflow.keras.layers import Dense, Input, Multiply, Concatenate  # 全连接层dense，输入层input，乘法层multiply，连接层concatenate
from tensorflow.keras.optimizers import Adam  # 用于训练模型时更新权重

from FuzzyLayer import FuzzyLayer  # 导入FuzzyLayer程序

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# plot diagnostic learning curves
def summarise_diagnostics(history): #定义函数，history是包含模型训练过程中的历史信息的参数
    # plot accuracy
    pyplot.subplot(211)  # 设置2x1的网格，并选择第一个子图（位于2x1网格的第一行）作为绘图区域
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')  # 绘制准确率曲线，蓝色线条，图例为train
    pyplot.xlabel('Epoch')  # 训练轮次
    pyplot.ylabel('Accuracy')

    # plot loss
    pyplot.subplot(212)
    pyplot.title('Mean Squared Error Loss')
    pyplot.plot(history.history['loss'], color='orange', label='train')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')

    pyplot.tight_layout() # 自动调整子图的布局
    pyplot.show() # 显示图像

import pickle
# pickle 是 Python 标准库中的模块，用于实现序列化和反序列化数据。序列化是指将数据结构或对象转换为字节流的过程

def load_dataset():
    # 定义列表名称
    train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_batch = 'test_batch'
    # 初始化，空集
    train_data = []
    train_labels = []

    # Load training data and labels
    for batch in train_batches:  # 遍历每个训练数据文件名
        with open(batch, 'rb') as f: # 用二进制只读模式打开每个训练文件
            batch_data = pickle.load(f, encoding='bytes') # 用pickle模块从文件中加载数据，确保正确读取字节型数据
            train_data.append(batch_data[b'data'])
            train_labels.append(batch_data[b'labels']) # 分别将当前训练数据和标签添加到相应的列表中

    trainX = np.concatenate(train_data)   # 将训练数据合并成大的numpy数组
    trainY = np.concatenate(train_labels)

    # Load test data and labels
    with open(test_batch, 'rb') as f: # 同上，打开测试数据文件
        test_data = pickle.load(f, encoding='bytes')
        testX = test_data[b'data']
        testY = test_data[b'labels']

    # Reshape and normalize data
    trainX = trainX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
    testX = testX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
    # -1表示根据数组的大小自动计算该轴的长度，重新转换成四维数组，3表示RGB三通道，32和32表示高度和宽度都是32像素，32x32
    # transpose是转换数组形状，在深度学习中，图像数据格式是（样本数，高度，宽度，通道数）
    # float32转换为32位浮点数，因为神经网络通常使用32位浮点数进行计算
    # /255.0 将RGB范围从【0，255】转换为【0，1】，便于训练

    # Convert labels to one-hot encoding，独热编码，只有一个元素为1，其他都是0
    trainY = to_categorical(trainY, num_classes=10)
    testY = to_categorical(testY, num_classes=10) # 总共10个类别

    return trainX, trainY, testX, testY
    # 返回处理后的训练数据、训练标签、测试数据和测试标签，即函数的输出结果。

def pre_process_data(train, test, keep_rgb=False): # 接受train和test的数据，以及可选参数keep_rgb，用于控制是否保留RGB格式
    # convert from integers to floats (To normalize correctly)
    train_norm = train.astype('float32')
    test_norm = test.astype('float32') # 转换为浮点型
    # normalize to range 0-1 【0，255】转换成【0，1】
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    # Keep RGB or convert it to Grey Image
    if keep_rgb: # ture,即为RGB格式
        # Flatten the (32x3) dimensions to (96) i.e. Flatten the image for Dense layer
        colour_processed_train = train_norm.reshape(*train_norm.shape[:2], -1)
        # train_norm.shape现在是（样本数，32，32，3）
        # [ : 2]表示切片操作，取出前两个维度，即（样本数，32）
        # 之后将数据展平，-1所代表的未知维度就是 样本数*32*32*3/样本数*32  （总的/已知的=未知的）
        colour_processed_test = test_norm.reshape(*test.shape[:2], -1)
        # 因为后面的全连接层需要的输入格式是一维向量，所以需要在这里变成合适的格式，即展平成一维向量

    else:
        # convert to grey-scale image  灰度图
        colour_processed_train = np.dot(train_norm[..., :3], [0.2989, 0.5870, 0.1140]) # 权重
        colour_processed_test = np.dot(test_norm[..., :3], [0.2989, 0.5870, 0.1140])

    # Swap 1st Dimension of the data i.e. number of examples with 2nd Dimension i.e. Number of rows in an image
    # This is important to fetch the Membership function as it takes an input vector (1x32) and we have a matrix (32x32)
    x_train_swap = np.einsum('kli->lki', colour_processed_train)
    x_test_swap = np.einsum('kli->lki', colour_processed_test)

    # 输入数组`colour_processed_train`的原始维度顺序为`(k, l, i)`，其中`k`表示样本数量，`l`表示图像的行数，`i`表示图像的列数。
    # 通过使用`'kli->lki'`作为参数，`np.einsum()`函数将输入数组的维度顺序从`(k, l, i)`转换为`(l, k, i)`，即将样本数量的维度移动到第二个维度位置。
    #
    # 这样的维度交换操作是为了适应后续操作中所使用的“Membership function”所需的输入向量格式，它期望输入为一个形状为`(l, i)`的矩阵，
    # 其中`l`表示输入向量的长度，`i`表示输入向量的维度。

    # Since "Membership function" feeds an input vector we convert our data to list of vectors
    # 32 lists of (32 size) vector makes it a single training/testing example or an image
    x_train_multi_input = [x for x in x_train_swap]
    x_test_multi_input = [x for x in x_test_swap] # 包含32个向量，每个向量32维的

    # Return List of train/test examples
    return x_train_multi_input, x_test_multi_input # 返回一个列表




def prepare_model(input_len, input_shape, num_classes, parameters):
    fuzz_membership_layer = []
    model_inputs = []
    for vector in range(input_len):



        model_inputs.append(Input(shape=(input_shape,)))
        # Membership Function layer
        fuzz_membership_layer.append(FuzzyLayer(parameters.membership_layer_units)(model_inputs[vector]))
        # 输出维度为5 的输出结果 添加到列表中
        #FuzzyLayer(parameters.membership_layer_units)`表示创建一个具有指定输出维度的模糊逻辑成员函数层。
        # 然后，`(model_inputs[vector])`表示将`model_inputs`中的第`vector`个输入向量作为该模糊逻辑成员函数层的输入
        # 存储所有输入向量对应的模糊逻辑成员函数层的输出结果

















    # Fuzzy Rule Layer
    rule_layer = Multiply()(fuzz_membership_layer)

    inp = Concatenate()(model_inputs)



    # Input DR Layers
    dr_layer_1 = Dense(parameters.dr_layer_1_units, activation='sigmoid')(inp)
    # parameters.dr_layer_1_units 表示该层神经元的数量，即输出张量的维度
    # 用sigmoid激活函数来模拟，inp为输入
    dr_layer_2 = Dense(parameters.dr_layer_2_units, activation='sigmoid')(dr_layer_1)

    # Fusion Layer
    fusion_layer = Concatenate()([rule_layer, dr_layer_2])

    # Fusion DR Layer
    fusion_dr_layer = Dense(parameters.fusion_dr_layer_units, activation='sigmoid')(fusion_layer)
    # 输出一个 benchsize * parameters.fusion_dr_layer_units



    # Task Driven Layer
    out = Dense(num_classes, activation='softmax')(fusion_dr_layer)


    model = Model(model_inputs, out)  # Model是keras中的模型类，用输入层和输出层构建完整的神经网络
    # compile model
    opt = Adam(learning_rate=parameters.learning_rate)  # 优化
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    return model


def parse_cli_parameters():
    parser = argparse.ArgumentParser(description="FuzzyDNN on CIFAR-10")
    parser.add_argument('--learning-rate', dest='learning_rate', default=10 ** -3, type=float, # ** ^
                        help='Learning Rate of your classifier. Default 0.001')
    parser.add_argument('--epoch', dest='epochs', default=100, type=int,
                        help='Number of times you want to train your data. Default 100')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16,
                        help='Batch size for prediction. Default=16.')
    parser.add_argument('--colour-image', dest='is_colour_image', action="store_true", default=False,
                        help='Passing this argument will keep the coloured image (RGB) during training. Default=False.')
    parser.add_argument('--membership-layer-units', dest='membership_layer_units', type=int, default=100,
                        help='Defines the number of units/nodes in the Membership Function Layer')
    parser.add_argument('--first-dr-layer-units', dest='dr_layer_1_units', type=int, default=100,
                        help='Defines the number of units in the first DR Layer')
    parser.add_argument('--second-dr-layer-units', dest='dr_layer_2_units', type=int, default=100,
                        help='Defines the number of units in the second DR Layer')
    parser.add_argument('--fusion-dr-layer-units', dest='fusion_dr_layer_units', type=int, default=100,
                        help='Defines the number of units in the Fusion DR Layer')
    parser.add_argument('--hide-graph', dest='should_hide_graph', action="store_true", default=False,
                        help='Hides the graph of results displayed via matplotlib')

    options = parser.parse_args()

    print("Starting with the following options:")
    pprint(vars(options))
    # 以下是 `ArgumentParser` 类的一些常用方法：
    #
    # - `add_argument`：添加命令行参数。
    # - `parse_args`：解析命令行参数，返回解析后的结果对象。
    # - `set_defaults`：设置参数的默认值。
    # - `add_argument_group`：创建一个参数组，用于对相关的参数进行分组。
    # - `error`：自定义错误消息的处理方法。
    return options


def main():
    cli_parameters = parse_cli_parameters()
    X_train, y_train, X_test, y_test = load_dataset()
    # X_train : (50000, 32, 32, 3)
    X_train, X_test = pre_process_data(X_train, X_test, keep_rgb=cli_parameters.is_colour_image)
    #X_train : (50000, 32，32) list
    # 50000个样本，32个vectors


    # 数据预处理

    # Defines the number of classes/categories and output vectors
    num_classes = y_test.shape[-1]
    # 分类的种类

    # Defines the number of input vectors
    input_length = len(X_train)
    # 32 第一个

    # Defines the shape of input layer
    input_shape = X_train[0].shape[-1]
    # 32 第二个




    #         [[1],[2],[3]] 这个张量的shape为（3,1）
    #         [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]]这个张量的shape为（3,2,2）,
    #         [1,2,3,4]这个张量的shape为（4，）
    #


    # input_shape：即张量的shape。从前往后对应由外向内的维度。
    #
    # input_length：代表序列长度，可以理解成有多少个样本
    #
    # input_dim：代表张量的维度，（很好理解，之前3个例子的input_dim分别为2,3,1）


    model = prepare_model(input_length, input_shape, num_classes, cli_parameters)


    # fit model
    history = model.fit(X_train, y_train, epochs=cli_parameters.epochs, batch_size=cli_parameters.batch_size)
    print('Evaluating the model')
    _, acc = model.evaluate(X_test, y_test)

    print('Model Evaluation Accuracy: {}'.format(acc))

    if not cli_parameters.should_hide_graph:
        summarise_diagnostics(history)


if __name__ == '__main__':
    main()
