import tensorflow as tf


class Dense(tf.Module):
    def __init__(self, in_dim, out_dim, name=None):
        super(Dense, self).__init__(name)
        with self.name_scope:
            self.weight = tf.Variable(
                initial_value=tf.random.normal((in_dim, out_dim),0.0,tf.sqrt(2.0/(in_dim+out_dim))),
                trainable=True,
                name="weight")
            self.bias = tf.Variable(
                initial_value=tf.zeros((out_dim, )),
                trainable=True,
                name="bias")

    def __call__(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias


class Conv2D(tf.Module):
    def __init__(self, k_size, in_dim, out_dim, strides=(1,1), padding="SAME", dilations=None, name=None):
        super(Conv2D, self).__init__(name)
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        with self.name_scope:
            self.weight = tf.Variable(
                initial_value=tf.random.normal(k_size + (in_dim, out_dim), 0.0, tf.sqrt(2.0/(in_dim+out_dim))),
                trainable=True,
                name="weight")
            self.bias = tf.Variable(
                initial_value=tf.zeros((out_dim, )),
                trainable=True,
                name="bias")

    def __call__(self, inputs):
        return tf.nn.conv2d(inputs, self.weight,strides=self.strides,padding=self.padding,dilations=self.dilations) + self.bias


class BatchNormalization(tf.Module):
    def __init__(self, feature_dim, momentum=0.99, epsilon=0.001, name=None):
        super(BatchNormalization, self).__init__(name)
        self.momentum = momentum
        self.epsilon = epsilon
        with self.name_scope:
            self.var = tf.Variable(
                initial_value=tf.ones((feature_dim,)),
                trainable=False,
                name="var")
            self.mean = tf.Variable(
                initial_value=tf.zeros((feature_dim,)),
                trainable=False,
                name="mean")
            self.gamma = tf.Variable(
                initial_value=tf.ones((feature_dim,)),
                trainable=True,
                name="gamma")
            self.beta = tf.Variable(
                initial_value=tf.zeros((feature_dim,)),
                trainable=True,
                name="beta")

    def __call__(self, inputs, training):

        if training:
            axis = tf.range(tf.rank(inputs))[:-1]
            batch_mean = tf.reduce_mean(inputs, axis)
            batch_var = tf.math.reduce_variance(inputs, axis)
            y = (inputs - batch_mean) * tf.math.rsqrt(batch_var + self.epsilon)
            y = self.gamma * y + self.beta
            
            new_mean = self.momentum*self.mean + (1.0-self.momentum)*batch_mean
            new_var  = self.momentum*self.var  + (1.0-self.momentum)*batch_var
            self.mean.assign(new_mean)
            self.var.assign(new_var)
        else:
            y = (inputs - self.mean) * tf.math.rsqrt(self.var + self.epsilon)
            y = self.gamma * y + self.beta
        
        return y
