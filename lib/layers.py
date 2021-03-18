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