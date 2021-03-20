import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from lib.optimizers import Adam
from lib.utils import set_gpu
from lib.models import LeNet5, LeNet_300_100, VGG10, VGG16
import math
import numpy as np


def train(model, optimizer, inputs, labels, batch_size):
    assert inputs.shape[0] == labels.shape[0]

    @tf.function
    def train_on_batch(batch_inputs, batch_labels):
        batch_preds = model(batch_inputs)
        batch_loss = tf.losses.categorical_crossentropy(batch_labels, batch_preds)
        batch_acc = tf.metrics.categorical_accuracy(batch_labels, batch_preds)
        gradients = tf.gradients(batch_loss, model.trainable_variables)
        optimizer(gradients)
        return batch_loss, batch_acc

    shuffle_idx = np.random.permutation(range(inputs.shape[0]))
    train_loss, train_acc = [], []
    for i in range(math.ceil(inputs.shape[0] / batch_size)):
        batch_idx = shuffle_idx[i * batch_size:(i + 1) * batch_size]
        batch_loss, batch_acc = train_on_batch(inputs[batch_idx], labels[batch_idx])
        train_loss.append(batch_loss)
        train_acc.append(batch_acc)
        print("\rtrain step {}, train_loss: {:.4}, train_acc: {:.4}".format(i + 1, np.mean(batch_loss), np.mean(batch_acc)), end="")

    return np.mean(np.concatenate(train_loss)), np.mean(np.concatenate(train_acc))


def evaluate(model, inputs, labels, batch_size):
    assert inputs.shape[0] == labels.shape[0]

    @tf.function
    def evaluate_on_batch(batch_inputs, batch_labels):
        batch_preds = model(batch_inputs)
        batch_loss = tf.losses.categorical_crossentropy(batch_labels, batch_preds)
        batch_acc = tf.metrics.categorical_accuracy(batch_labels, batch_preds)
        return batch_loss, batch_acc

    evaluate_loss, evaluate_acc = [], []
    for i in range(math.ceil(inputs.shape[0]/batch_size)):
        batch_loss, batch_acc = evaluate_on_batch(inputs[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])
        evaluate_loss.append(batch_loss)
        evaluate_acc.append(batch_acc)
        print("\revaluate step {}, evaluate_loss: {:.4}, evaluate_acc: {:.4}".format(i + 1, np.mean(batch_loss), np.mean(batch_acc)), end="")

    return np.mean(np.concatenate(evaluate_loss)), np.mean(np.concatenate(evaluate_acc))


if __name__ == "__main__":
    set_gpu(0)
    batch_size = 128
    num_classes = 10
    epochs = 100
    val_percent = 0.9
    dataset_model = "mnist_lenet_300_100"

    if dataset_model == "mnist_lenet_300_100":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        image_shape = (784, )
        model = LeNet_300_100(image_shape[0], num_classes)
    elif dataset_model == "mnist_lenet5":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        image_shape = (28, 28, 1)
        model = LeNet5(image_shape, num_classes)
    elif dataset_model == "mnist_vgg10":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        image_shape = (28, 28, 1)
        model = VGG10(image_shape, num_classes)
    elif dataset_model == "cifar10_vgg16":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        image_shape = (32, 32, 3)
        model = VGG16(image_shape, num_classes)
    else:
        raise ValueError("Unknown dataset and model: {}.".format(dataset_model))

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train.reshape((-1, ) + image_shape)
    x_test = x_test.reshape((-1, ) + image_shape)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    num_train = x_train.shape[0]
    x_train, x_val = x_train[:int(val_percent*num_train)], x_train[int(val_percent*num_train):]
    y_train, y_val = y_train[:int(val_percent*num_train)], y_train[int(val_percent*num_train):]

    optimizer = Adam(model.trainable_variables, lr=1e-3)

    for e in range(epochs):
        train_loss, train_acc = train(model, optimizer, x_train, y_train, batch_size)
        val_loss, val_acc = evaluate(model, x_val, y_val, batch_size)
        test_loss, test_acc = evaluate(model, x_test, y_test, batch_size)

        print("\repoch {}, train_loss: {:.4}, train_acc: {:.4}, val_loss: {:.4}, val_acc: {:.4}, test_loss: {:.4}, test_acc: {:.4}".format(
            e + 1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))