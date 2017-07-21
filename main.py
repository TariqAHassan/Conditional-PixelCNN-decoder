import tensorflow as tf
import numpy as np
from models import PixelCNN
from autoencoder import *
from utils import *
from tqdm import tqdm


def train(conf, data):
    X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channels])
    model = PixelCNN(X, conf)

    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(model.loss)

    clipped_gradients = [(tf.clip_by_value(i[0], -conf.grad_clip, conf.grad_clip), i[1]) for i in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if os.path.exists(conf.ckpt_file):
            saver.restore(sess, conf.ckpt_file)
            print("Model Restored")

        if conf.epochs > 0:
            print("Started Model Training...")

        pointer, cost = 0, np.nan
        for i in tqdm(range(conf.epochs), desc='Epoch'):
            print("Epoch: %d, Cost: %f" % (i, cost))
            for j in tqdm(range(conf.num_batches), desc='Iteration', leave=False):
                if conf.data == "mnist":
                    batch_X, batch_y = data.train.next_batch(conf.batch_size)
                    batch_X = binarize(batch_X.reshape(
                        [conf.batch_size, conf.img_height, conf.img_width, conf.channels]))
                    batch_y = one_hot(batch_y, conf.num_classes)
                else:
                    batch_X, pointer = get_batch(data, pointer, conf.batch_size)
                data_dict = {X: batch_X}
                if conf.conditional is True:
                    data_dict[model.h] = batch_y
                _, cost = sess.run([optimizer, model.loss], feed_dict=data_dict)

            if (i + 1) % 10 == 0:
                saver.save(sess, conf.ckpt_file)
                generate_samples(sess, X, model.h, model.pred, conf, "")

        generate_samples(sess, X, model.h, model.pred, conf, "")


class Args(object):
    def __init__(self, args_dict):
        self.__dict__.update(args_dict)


if __name__ == "__main__":
    args_dict = {
        'data': 'mnist',
        'layers': 12,
        'f_map': 32,
        'epochs': 50,
        'batch_size': 100,
        'grad_clip': 1,
        'model': '',
        'data_path': 'data',
        'ckpt_path': 'ckpts',
        'samples_path': 'samples',
        'summary_path': 'logs'
    }
    conf = Args(args_dict)

    if conf.data == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data

        if not os.path.exists(conf.data_path):
            os.makedirs(conf.data_path)
        data = input_data.read_data_sets(conf.data_path)
        conf.num_classes = 10
        conf.img_height = 28
        conf.img_width = 28
        conf.channels = 1
        conf.num_batches = data.train.num_examples // conf.batch_size
    else:
        from keras.datasets import cifar10

        data = cifar10.load_data()
        labels = data[0][1]
        data = data[0][0].astype(np.float32)
        data[:, 0, :, :] -= np.mean(data[:, 0, :, :])
        data[:, 1, :, :] -= np.mean(data[:, 1, :, :])
        data[:, 2, :, :] -= np.mean(data[:, 2, :, :])
        data = np.transpose(data, (0, 2, 3, 1))
        conf.img_height = 32
        conf.img_width = 32
        conf.channels = 3
        conf.num_classes = 10
        conf.num_batches = data.shape[0] // conf.batch_size

    conf = makepaths(conf)
    if conf.model == '':
        conf.conditional = False
        train(conf, data)
    elif conf.model.lower() == 'conditional':
        conf.conditional = True
        train(conf, data)
    elif conf.model.lower() == 'autoencoder':
        conf.conditional = True
        trainAE(conf, data)
