import tensorflow as tf
from lib.layers import Dense


class LeNet_300_100(tf.Module):
    def __init__(self, input_dim, output_dim, name="lenet_300_100"):
        super(LeNet_300_100, self).__init__(name)
        self.dense1 = Dense(input_dim,300,"dense1")
        self.dense2 = Dense(300,100,"dense2")
        self.classifier = Dense(100,output_dim,"dense2")

    def __call__(self, inputs):
        y = self.dense1(inputs)
        y = tf.nn.relu(y)
        y = self.dense2(y)
        y = tf.nn.relu(y)
        y = self.classifier(y)
        y = tf.nn.softmax(y,axis=-1)
        return y