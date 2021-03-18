import tensorflow as tf


def set_gpu(gpu_idx):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if gpu_idx == None:
                tf.config.experimental.set_visible_devices([], 'GPU')
            else:
                tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
                # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)