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


class VGG10(tf.Module):
    def __init__(self, image_shape, num_classes, name="vgg16"):
        super(VGG10, self).__init__(name)
        height, width, channel = image_shape
        self.block1 = Conv2DMaxPoolBlock(channel, 64, 2, "block1")
        self.block2 = Conv2DMaxPoolBlock(64,  128, 2, "block2")
        self.block3 = Conv2DMaxPoolBlock(128, 256, 3, "block3")

        self.dense1 = Dense(256*(height//8)*(width//8), 512, "dense1")
        self.dense2 = Dense(512, 512, "dense2")
        self.dense3 = Dense(512, num_classes, "dense3")

    def __call__(self, inputs):
        batch_size = tf.shape(inputs)[0]

        y = self.block1(inputs)
        y = self.block2(y)
        y = self.block3(y)

        y = tf.reshape(y, (batch_size, -1))
        y = self.dense1(y)
        y = tf.nn.relu(y)
        y = self.dense2(y)
        y = tf.nn.relu(y)
        y = self.dense3(y)
        y = tf.nn.softmax(y,axis=-1)
        return y


class VGG16(tf.Module):
    def __init__(self, image_shape, num_classes, name="vgg16"):
        super(VGG16, self).__init__(name)
        height, width, channel = image_shape
        self.block1 = Conv2DMaxPoolBlock(channel, 64, 2, "block1")
        self.block2 = Conv2DMaxPoolBlock(64,  128, 2, "block2")
        self.block3 = Conv2DMaxPoolBlock(128, 256, 3, "block3")
        self.block4 = Conv2DMaxPoolBlock(256, 512, 3, "block4")
        self.block5 = Conv2DMaxPoolBlock(512, 512, 3, "block5")

        self.dense1 = Dense(512*(height//32)*(width//32), 4096, "dense1")
        self.dense2 = Dense(4096, 4096, "dense2")
        self.dense3 = Dense(4096, num_classes, "dense3")

    def __call__(self, inputs):
        batch_size = tf.shape(inputs)[0]

        y = self.block1(inputs)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)

        y = tf.reshape(y, (batch_size, -1))
        y = self.dense1(y)
        y = tf.nn.relu(y)
        y = self.dense2(y)
        y = tf.nn.relu(y)
        y = self.dense3(y)
        y = tf.nn.softmax(y,axis=-1)
        return y


class Conv2DMaxPoolBlock(tf.Module):
    def __init__(self, in_filters, out_filters, num_conv2d, name):
        assert num_conv2d >= 1
        super(Conv2DMaxPoolBlock, self).__init__(name)
        self.num_conv2d = num_conv2d
        self.convs = [
            Conv2D(
                k_size=(3, 3),
                in_dim=in_filters if i == 0 else out_filters,
                out_dim=out_filters,
                name="{}_conv{}".format(name, i + 1)
            )
            for i in range(self.num_conv2d)
        ]

    def __call__(self, inputs):
        y = inputs
        for conv in self.convs:
            y = conv(y)
            y = tf.nn.relu(y)
        y = tf.nn.max_pool(y,(2,2),(2,2),"VALID")
        return y