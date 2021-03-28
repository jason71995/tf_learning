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


class BatchNormalization(tf.Module):

    def __init__(self, feature_dim, momentum=0.9, name=None):
        super(BatchNormalization, self).__init__(name)
        self.momentum = momentum

        self.std = tf.Variable(
            initial_value=tf.ones((feature_dim,)),
            trainable=False,
            name="{}_std".format(name))
        self.mean = tf.Variable(
            initial_value=tf.zeros((feature_dim,)),
            trainable=False,
            name="{}_mean".format(name))
        self.gamma = tf.Variable(
            initial_value=tf.ones((feature_dim,)),
            trainable=True,
            name="{}_gamma".format(name))
        self.beta = tf.Variable(
            initial_value=tf.zeros((feature_dim,)),
            trainable=True,
            name="{}_beta".format(name))

    def __call__(self, inputs, training):

        if training:
            in_rank = tf.rank(inputs)
            axis = tf.range(in_rank)[:-1]
            in_mean = tf.reduce_mean(inputs, axis)
            in_std = tf.math.reduce_std(inputs, axis)
            self.mean.assign((1.0-self.momentum)*in_mean+self.momentum*self.mean)
            self.std.assign((1.0-self.momentum)*in_std+self.momentum*self.std)

        y = (inputs-self.mean)/(self.std+1e-7)
        y = self.gamma*y+self.beta
        return y


class Dropout(tf.Module):
    def __init__(self, drop_rate, name=None):
        super(Dropout, self).__init__(name)
        self.drop_rate = drop_rate

    def __call__(self, inputs, training):
        if training:
            # return tf.nn.dropout(inputs, self.rate)
            mask = tf.cast(tf.random.uniform((tf.shape(inputs)[-1], ), 0.0, 1.0) > self.drop_rate, "float32")
            return mask * inputs / (1.0-self.drop_rate)

        return inputs
