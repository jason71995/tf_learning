import tensorflow as tf


class Dense(tf.Module):
    def __init__(self, in_dim, out_dim, name=None):
        super(Dense, self).__init__(name)
        self.weight = tf.Variable(
            initial_value=tf.random.normal((in_dim, out_dim),0.0,tf.sqrt(2.0/(in_dim+out_dim))),
            trainable=True,
            name="{}_weight".format(name))
        self.bias = tf.Variable(
            initial_value=tf.zeros((out_dim, )),
            trainable=True,
            name="{}_bias".format(name))

    def __call__(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias


class Conv2D(tf.Module):
    def __init__(self, k_size, in_dim, out_dim, strides=(1,1), padding="SAME", dilations=None, name=None):
        super(Conv2D, self).__init__(name)
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.weight = tf.Variable(
            initial_value=tf.random.normal(k_size + (in_dim, out_dim), 0.0, tf.sqrt(2.0/(in_dim+out_dim))),
            trainable=True,
            name="{}_weight".format(name))
        self.bias = tf.Variable(
            initial_value=tf.zeros((out_dim, )),
            trainable=True,
            name="{}_bias".format(name))

    def __call__(self, inputs):
        return tf.nn.conv2d(inputs, self.weight,strides=self.strides,padding=self.padding,dilations=self.dilations) + self.bias