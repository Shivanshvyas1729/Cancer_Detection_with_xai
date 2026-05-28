import tensorflow as tf
from tensorflow.keras import layers, Model

# -------- Avg2Max Pooling --------

@tf.keras.utils.register_keras_serializable()
class Avg2MaxPooling(layers.Layer):
  """Novel Avg2Max Pooling layer """
  def __init__(self,pool_size=3,strides=2,padding='same',**kwargs):
    super(Avg2MaxPooling,self).__init__(**kwargs)
    
    self.pool_size = pool_size
    self.strides = strides
    self.padding = padding

    self.avg_pool = layers.AveragePooling2D(pool_size,strides,padding)
    self.max_pool = layers.MaxPooling2D(pool_size,strides,padding)
    self.bn = layers.BatchNormalization()


  def call(self,inputs):
    x=self.avg_pool(inputs) -2*self.max_pool(inputs)
    return self.bn(x)


def get_config(self):
    config = super(Avg2MaxPooling, self).get_config()
    config.update({
        "pool_size": self.pool_size,
        "strides": self.strides,
        "padding": self.padding
    })
    return config



# -------- Depthwise Separable Conv --------
@tf.keras.utils.register_keras_serializable()
class DepthwiseSeparableConv(layers.Layer):
    """Depthwise Separable Convolution with ReLU"""
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super(DepthwiseSeparableConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        
    def build(self, input_shape):
        self.dw = layers.DepthwiseConv2D(self.kernel_size, self.strides, padding='same')
        self.pw = layers.Conv2D(self.filters, 1, strides=1)
        self.bn = layers.BatchNormalization()
        super(DepthwiseSeparableConv , self).build(input_shape)

    def call(self, inputs):
        x = self.dw(inputs)
        x = self.pw(x)
        return tf.nn.relu(self.bn(x))

    def get_config(self):
        config = super(DepthwiseSeparableConv, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
        })
        return config

# -------- FibonacciNet (Notebook4e) --------



def create_fibonacci_net(input_shape=(224,224,3),num_classes=1):

    inputs= layers.Input(shape=input_shape)




  # block1 (21 filter)
    x= layers.Conv2D(21,3,padding='same')(inputs) #(224, 224, 21)

    x= layers.BatchNormalization()(x)  #(224, 224, 21)

    x= layers.ReLU()(x)

    x= layers.MaxPooling2D(2)(x)  # 112x112x21


    # Block 2 34 filters

    x = layers.Conv2D(34,3,padding='same')(x) # 112x112x34
    x = layers.BatchNormalization()(x) # 112x112x34
    x = layers.ReLU()(x)# 112x112x34
    x2 = x  # 112x112x34
    x= layers.MaxPooling2D(2)(x)  # 56x56x34

    # Block 3  (55 filters)

    x = layers.Conv2D(55,3,padding='same')(x)  # 56x56x55
    x = layers.BatchNormalization()(x) # 56x56x55
    x = layers.ReLU()(x) # 56x56x55
    x3=x

    x= layers.MaxPooling2D(2)(x)  # 28x28x55


    # =================================================
    # pcb1: Block 2 -> Block 4
    # =================================================

    pcb1 = layers.Conv2D(24,3,padding='same')(x2)  # 112x112x24
    pcb1  = Avg2MaxPooling()(pcb1) #w/s  (56×56×24)


    pcb1 = layers.Conv2D(24,3,padding='same')(pcb1) #(56×56×24)
    pcb1  = Avg2MaxPooling()(pcb1) #(28×28×24)


    # =================================================
    # Block 4
    # =================================================

    x = layers.Conv2D(89,3,padding='same')(x)  # 28x28x89
    x= layers.BatchNormalization()(x) # 28x28x89
    x= layers.ReLU()(x) # 28x28x89

    x=layers.MaxPooling2D(2)(x) #  # 14x14x89


    pcb1 = layers.Resizing(14,14)(pcb1)

    x= layers.Concatenate()([x,pcb1]) #14×14×113



    # =================================================
    # pcb2 : Block3 -> Block5
    # =================================================


    pcb2 =layers.Conv2D(24,3,padding='same')(x3) # # 56x56x24

    pcb2 = Avg2MaxPooling()(pcb2)#  (28x28x24)


    pcb2 =layers.Conv2D(24,3,padding='same')(pcb2) #  (28x28x24)
    pcb2 = Avg2MaxPooling()(pcb2)# (14,14,24)


    # =================================================
    # Block 5
    # =================================================

    x = layers.Conv2D(144,3,padding='same')(x)  # 14x14x114

    x= layers.BatchNormalization()(x) #  14x14x114

    x= layers.ReLU()(x) # 14x14x114

    x=layers.MaxPooling2D(2)(x) # 7x7x144


    pcb2 = layers.Resizing(7,7)(pcb2)

    x = layers.Concatenate()([x,pcb2]) #(7x7x168)


# -------- Block 6 (233 filters, DWSC) --------


    x =layers.Conv2D(233,3,padding='same')(x) # 7x7x233

    x= layers.BatchNormalization()(x) # 7x7x233

    x= layers.ReLU()(x) # 7x7x233


    x  = DepthwiseSeparableConv(233)(x)

# ----------------------------------------------

# -------- Block 7 (377 filters, DWSC) --------



    x =layers.Conv2D(377,3,padding='same')(x) # 7x7x377

    x= layers.BatchNormalization()(x) #7x7x 377

    x= layers.ReLU()(x) # 7x7x377


    x  = DepthwiseSeparableConv(377)(x)  # 7x7x377


    x = layers.GlobalAveragePooling2D()(x)

    #output

    outputs = layers.Dense(num_classes,activation='sigmoid')(x)

    return Model(inputs,outputs)



