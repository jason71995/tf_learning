from re import L
import tensorflow as tf
from lib.layers import Dense, Conv2D, BatchNormalization


class ResNet18(tf.Module):
    def __init__(self, image_shape, num_classes, name="resnet18"):
        super(ResNet18, self).__init__(name)

        with self.name_scope:
            self.conv1 = Conv2D(k_size=(7, 7), in_dim=image_shape[-1], out_dim=64, strides=(2,2), name="conv1")
            self.bn1 = BatchNormalization(64, name="bn1")

            self.blocks = [
                ResBlockConv2(64, 64, (1, 1), "s1_b1"),
                ResBlockConv2(64, 64, (1, 1), "s1_b2"),

                ResBlockConv2(64, 128, (2, 2), "s2_b1"),
                ResBlockConv2(128, 128, (1, 1), "s2_b2"),

                ResBlockConv2(128, 256, (2, 2), "s3_b1"),
                ResBlockConv2(256, 256, (1, 1), "s3_b2"),

                ResBlockConv2(256, 512, (2, 2), "s4_b1"),
                ResBlockConv2(512, 512, (1, 1), "s4_b2"),
            ]

            self.dense = Dense(512, num_classes, "dense")

    def __call__(self, inputs, training):
        y = self.conv1(inputs)
        y = self.bn1(y, training)
        y = tf.nn.relu(y)
        y = tf.nn.max_pool(y, (3, 3), (2, 2), "SAME")

        for block in self.blocks:
            y = block(y, training)

        # global pooling
        y = tf.reduce_mean(y, axis=[1,2])
        y = self.dense(y)
        y = tf.nn.softmax(y,axis=-1)
        return y


class ResNet50(tf.Module):
    def __init__(self, image_shape, num_classes, name="resnet50"):
        super(ResNet50, self).__init__(name)

        with self.name_scope:
            self.conv1 = Conv2D(k_size=(7, 7), in_dim=image_shape[-1], out_dim=64, strides=(2,2), name="conv1")
            self.bn1 = BatchNormalization(64, name="bn1")

            self.blocks = [
                ResBlockConv3(64,  256, (1, 1), "s1_b1"),
                ResBlockConv3(256, 256, (1, 1), "s1_b2"),
                ResBlockConv3(256, 256, (1, 1), "s1_b3"),

                ResBlockConv3(256, 512, (2, 2), "s2_b1"),
                ResBlockConv3(512, 512, (1, 1), "s2_b2"),
                ResBlockConv3(512, 512, (1, 1), "s2_b3"),
                ResBlockConv3(512, 512, (1, 1), "s2_b4"),

                ResBlockConv3(512,  1024, (2, 2), "s3_b1"),
                ResBlockConv3(1024, 1024, (1, 1), "s3_b2"),
                ResBlockConv3(1024, 1024, (1, 1), "s3_b3"),
                ResBlockConv3(1024, 1024, (1, 1), "s3_b4"),
                ResBlockConv3(1024, 1024, (1, 1), "s3_b5"),
                ResBlockConv3(1024, 1024, (1, 1), "s3_b6"),

                ResBlockConv3(1024, 2048, (2, 2), "s4_b1"),
                ResBlockConv3(2048, 2048, (1, 1), "s4_b2"),
                ResBlockConv3(2048, 2048, (1, 1), "s4_b3"),
            ]

            self.dense = Dense(2048, num_classes, "dense")

    def __call__(self, inputs, training):
        y = self.conv1(inputs)
        y = self.bn1(y, training)
        y = tf.nn.relu(y)
        y = tf.nn.max_pool(y, (3, 3), (2, 2), "SAME")

        for block in self.blocks:
            y = block(y, training)

        # global pooling
        y = tf.reduce_mean(y, axis=[1,2])
        y = self.dense(y)
        y = tf.nn.softmax(y,axis=-1)
        return y


class ResBlockConv2(tf.Module):
    def __init__(self, in_filters, out_filters, strides, name):
        super(ResBlockConv2, self).__init__(name)

        self.is_short_cut = in_filters != out_filters
        
        with self.name_scope:
            if self.is_short_cut:
                self.conv0 = Conv2D(k_size=(1, 1), in_dim=in_filters, out_dim=out_filters, strides=strides, name="conv0")
                self.bn0 = BatchNormalization(out_filters,name="bn0")

            self.conv1 = Conv2D(k_size=(1, 1), in_dim=in_filters, out_dim=out_filters, strides=strides, name="conv1")
            self.bn1 = BatchNormalization(out_filters,name="bn1")
            self.conv2 = Conv2D(k_size=(3, 3), in_dim=out_filters, out_dim=out_filters, name="conv2")
            self.bn2 = BatchNormalization(out_filters,name="bn2")

    def __call__(self, inputs, training):

        if self.is_short_cut:
            short_cut = self.conv0(inputs)
            short_cut = self.bn0(short_cut, training)
        else:
            short_cut = inputs

        y = self.conv1(inputs)
        y = self.bn1(y, training)
        y = tf.nn.relu(y)

        y = self.conv2(y)
        y = self.bn2(y, training)
        y = y + short_cut
        y = tf.nn.relu(y)
        return y


class ResBlockConv3(tf.Module):
    def __init__(self, in_filters, out_filters, strides, name):
        super(ResBlockConv3, self).__init__(name)

        self.is_short_cut = in_filters != out_filters
        
        with self.name_scope:
            if self.is_short_cut:
                self.conv0 = Conv2D(k_size=(1, 1), in_dim=in_filters, out_dim=out_filters, strides=strides, name="conv0")
                self.bn0 = BatchNormalization(out_filters,name="bn0")

            self.conv1 = Conv2D(k_size=(1, 1), in_dim=in_filters, out_dim=out_filters//4, strides=strides, name="conv1")
            self.bn1 = BatchNormalization(out_filters//4,name="bn1")
            self.conv2 = Conv2D(k_size=(3, 3), in_dim=out_filters//4, out_dim=out_filters//4, name="conv2")
            self.bn2 = BatchNormalization(out_filters//4,name="bn2")
            self.conv3 = Conv2D(k_size=(1, 1), in_dim=out_filters//4, out_dim=out_filters, name="conv3")
            self.bn3 = BatchNormalization(out_filters,name="bn3")

    def __call__(self, inputs, training):

        if self.is_short_cut:
            short_cut = self.conv0(inputs)
            short_cut = self.bn0(short_cut, training)
        else:
            short_cut = inputs

        y = self.conv1(inputs)
        y = self.bn1(y, training)
        y = tf.nn.relu(y)

        y = self.conv2(y)
        y = self.bn2(y, training)
        y = tf.nn.relu(y)

        y = self.conv3(y)
        y = self.bn3(y, training)
        y = y + short_cut
        y = tf.nn.relu(y)
        return y
