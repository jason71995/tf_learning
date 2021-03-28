import tensorflow as tf


class SGD(tf.Module):
    def __init__(self,
                 params,
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=0.0,
                 name="sgd"):

        super(SGD, self).__init__(name)

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.step = 0
        self.exp_avg = []

        for param in params:
            self.exp_avg.append(tf.Variable(
                initial_value=tf.zeros(param.numpy().shape),
                trainable=False,
                name="{}_{}_exp_avg".format(name, param.name)))

    def __call__(self, grads, params):
        self.step += 1
        for i, (grad, param) in enumerate(zip(grads, params)):
            exp_avg = self.exp_avg[i]

            if self.weight_decay != 0:
                grad += self.weight_decay * param

            exp_avg.assign((1 - self.momentum) * grad + self.momentum * exp_avg)
            param.assign_add(-self.lr * exp_avg)


class Adam(tf.Module):
    def __init__(self,
                 params,
                 lr=1e-3,
                 beta1=0.9,
                 beta2=0.999,
                 weight_decay=0.0,
                 eps=1e-7,
                 amsgrad=False,
                 name="adam"):

        super(Adam, self).__init__(name)

        self.amsgrad = amsgrad
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.step = 0
        self.exp_avg = []
        self.exp_avg_sq = []

        if self.amsgrad:
            self.max_exp_avg_sq = []

        for param in params:
            self.exp_avg.append(tf.Variable(
                initial_value=tf.zeros(param.numpy().shape),
                trainable=False,
                name="{}_{}_exp_avg".format(name, param.name)))

            self.exp_avg_sq.append(tf.Variable(
                initial_value=tf.zeros(param.numpy().shape),
                trainable=False,
                name="{}_{}_exp_avg_sq".format(name, param.name)))

            if self.amsgrad:
                self.max_exp_avg_sq.append(tf.Variable(
                    initial_value=tf.zeros(param.numpy().shape),
                    trainable=False,
                    name="{}_{}_max_exp_avg_sq".format(name, param.name)))

    def __call__(self, grads, params):

        self.step += 1
        for i, (grad, param) in enumerate(zip(grads, params)):
            exp_avg = self.exp_avg[i]
            exp_avg_sq = self.exp_avg_sq[i]

            bias_correction1 = 1 - self.beta1 ** self.step
            bias_correction2 = 1 - self.beta2 ** self.step

            if self.weight_decay != 0:
                grad += self.weight_decay * param

            exp_avg.assign((1 - self.beta1) * grad + self.beta1 * exp_avg)
            exp_avg_sq.assign((1 - self.beta2) * grad * grad + self.beta2 * exp_avg_sq)

            if self.amsgrad:
                max_exp_avg_sq = self.max_exp_avg_sq[i]
                max_exp_avg_sq.assign(tf.maximum(max_exp_avg_sq, exp_avg_sq))
                denom = self.eps + tf.sqrt(max_exp_avg_sq) / tf.sqrt(bias_correction2)
            else:
                denom = self.eps + tf.sqrt(exp_avg_sq) / tf.sqrt(bias_correction2)

            step_size = self.lr / bias_correction1
            param.assign_add(-step_size * exp_avg / denom)