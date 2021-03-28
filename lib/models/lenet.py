import tensorflow as tf
from lib.layers import Dense, Conv2D


class LeNet_300_100(tf.Module):
    def __init__(self, input_dim, output_dim, name="lenet_300_100"):
        super(LeNet_300_100, self).__init__(name)
        self.dense1 = Dense(input_dim,300,"dense1")
        self.dense2 = Dense(300,100,"dense2")
        self.dense3 = Dense(100,output_dim,"dense3")

    def __call__(self, inputs):
        y = self.dense1(inputs)
        y = tf.nn.relu(y)
        y = self.dense2(y)
        y = tf.nn.relu(y)
        y = self.dense3(y)
        y = tf.nn.softmax(y,axis=-1)
        return y


class LeNet5(tf.Module):
    def __init__(self, image_shape, num_classes, name="lenet5"):
        super(LeNet5, self).__init__(name)
        height, width, channel = image_shape
        self.conv1 = Conv2D((3,3), channel, 6, padding="SAME", name="conv1")
        self.conv2 = Conv2D((3,3), 6, 16, padding="SAME", name="conv2")
        self.dense1 = Dense(16*(height//4)*(width//4), 120, "dense1")
        self.dense2 = Dense(120, 84, "dense2")
        self.dense3 = Dense(84, num_classes, "dense3")

    def __call__(self, inputs):
        batch_size = tf.shape(inputs)[0]

        y = self.conv1(inputs)
        y = tf.nn.relu(y)
        y = tf.nn.max_pool(y,(2,2),(2,2),"VALID")

        y = self.conv2(y)
        y = tf.nn.relu(y)
        y = tf.nn.max_pool(y,(2,2),(2,2),"VALID")

        y = tf.reshape(y, (batch_size, -1))
        y = self.dense1(y)
        y = tf.nn.relu(y)
        y = self.dense2(y)
        y = tf.nn.relu(y)
        y = self.dense3(y)
        y = tf.nn.softmax(y,axis=-1)
        return y