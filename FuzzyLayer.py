from keras import backend as K
from tensorflow.keras.layers import Layer # 自定义的模糊逻辑成员函数层

# 层类可以用于在神经网络模型中添加一个模糊逻辑成员函数层，以实现某种特定的功能或行为。


class FuzzyLayer(Layer):
   # 输出维度（数量），μ，σ，关键字参数
    def __init__(self, output_dim, initialiser_centers=None, initialiser_sigmas=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            # 将input_dim的值用于设置input_shape

        self.output_dim = output_dim
        self.initialiser_centers = initialiser_centers
        self.initialiser_sigmas = initialiser_sigmas
        super(FuzzyLayer, self).__init__(**kwargs) # 可以调用父类 `Layer` 的构造函数，并传递 `**kwargs` 参数。`**kwargs` 表示关键字参数集合，用于传递任意数量的关键字参数。

    def build(self, input_shape): # 构建权重
        self.fuzzy_degree = self.add_weight(name='fuzzy_degree',
                                            shape=(input_shape[-1], self.output_dim),
                                            initializer=self.initialiser_centers if self.initialiser_centers is not None else 'uniform',
                                            trainable=True)
        # self.add_weight是layer类提供的方法，用于在层中添加权重
        # input_shape[-1] 表示输入数据的最后一个维度的大小，即输入向量的长度
        # self.output_dim 表示成员函数层的输出维度，即成员函数的数量
        # 这个权重的形状是（长度，数量）

        self.sigma = self.add_weight(name='sigma',
                                     shape=(input_shape[-1], self.output_dim),
                                     initializer=self.initialiser_sigmas if self.initialiser_sigmas is not None else 'ones',
                                     trainable=True)
        super(FuzzyLayer, self).build(input_shape) # 调用父类函数

    def call(self, input, **kwargs):
        x = K.repeat_elements(K.expand_dims(input, axis=-1), self.output_dim, -1)#在最后一个维度上增加一个维度
        # (batch_size, input_dim, self.output_dim)
        # input (batch_size, input_dim) then  K.expand_dims(input, axis=-1) (batch_size, input_dim, 1) -1 意味着在最后加一个维度
        # 然后进行repeat 将 这个  (batch_size, input_dim, 1) 重复 self.output_dim 次 变成上面的式子， 方便计算，


        fuzzy_out = K.exp(-K.sum(K.square((x - self.fuzzy_degree) / (self.sigma ** 2)), axis=-2, keepdims=False))
        # 高斯隶属函数

        # 隶属度？
        return fuzzy_out  # 返回隶属度

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)
#       # 变成(batch_size, input_dim, self.output_dim)输出
