import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from lib.optimizers import Adam
from lib.utils import set_gpu
from lib.models import VGG16
import math
import numpy as np

if __name__ == "__main__":
    set_gpu(0)
    batch_size = 128
    num_classes = 10
    epochs = 100
    val_percent = 0.9
    image_shape = (32, 32, 3)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train.reshape((-1, ) + image_shape)
    x_test = x_test.reshape((-1, ) + image_shape)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    num_train = x_train.shape[0]
    x_train, x_val = x_train[:int(val_percent*num_train)], x_train[int(val_percent*num_train):]
    y_train, y_val = y_train[:int(val_percent*num_train)], y_train[int(val_percent*num_train):]

    model = VGG16(image_shape, num_classes)
    optimizer = Adam(model.trainable_variables, lr=1e-3)

    @tf.function
    def train(inputs, labels):
        preds = model(inputs)
        loss = tf.losses.categorical_crossentropy(labels, preds)
        acc = tf.metrics.categorical_accuracy(labels, preds)
        gradients = tf.gradients(loss, model.trainable_variables)
        optimizer(gradients)
        return loss, acc

    @tf.function
    def evaluate(inputs, labels):
        y_pred = model(inputs)
        loss = tf.losses.categorical_crossentropy(labels, y_pred)
        acc = tf.metrics.categorical_accuracy(labels, y_pred)
        return loss, acc

    for e in range(epochs):
        # ==================== train ====================
        shuffle_idx = np.random.permutation(range(x_train.shape[0]))
        train_loss, train_acc = [], []
        for i in range(math.ceil(x_train.shape[0]/batch_size)):
            batch_idx = shuffle_idx[i * batch_size:(i + 1) * batch_size]
            loss, acc = train(x_train[batch_idx], y_train[batch_idx])
            train_loss.append(loss)
            train_acc.append(acc)
            print("\rstep {}, train_loss: {:.4}, train_acc: {:.4}".format(
                i + 1, np.mean(loss), np.mean(acc)), end="")
        train_loss, train_acc = np.mean(np.concatenate(train_loss)), np.mean(np.concatenate(train_acc))

        # ==================== validation ====================
        val_loss, val_acc = [], []
        for i in range(math.ceil(x_val.shape[0]/batch_size)):
            loss, acc = evaluate(x_val[i * batch_size:(i + 1) * batch_size],
                                 y_val[i * batch_size:(i + 1) * batch_size])
            val_loss.append(loss)
            val_acc.append(acc)
        val_loss, val_acc = np.mean(np.concatenate(val_loss)), np.mean(np.concatenate(val_acc))

        # ==================== test ====================
        test_loss, test_acc = [], []
        for i in range(math.ceil(x_test.shape[0]/batch_size)):
            loss, acc = evaluate(x_test[i * batch_size:(i + 1) * batch_size],
                                 y_test[i * batch_size:(i + 1) * batch_size])
            test_loss.append(loss)
            test_acc.append(acc)
        test_loss, test_acc = np.mean(np.concatenate(test_loss)), np.mean(np.concatenate(test_acc))

        print("\repoch {}, train_loss: {:.4}, train_acc: {:.4}, val_loss: {:.4}, val_acc: {:.4}, test_loss: {:.4}, test_acc: {:.4}".format(
            e + 1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))