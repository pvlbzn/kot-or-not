import h5py
import algorithms

import numpy as np
from scipy import ndimage, misc

num_px = 0


def load_data():
    dataset = h5py.File('./data/train_catvnoncat.h5', 'r')

    data = np.array(dataset['train_set_x'][:])
    labels = np.array(dataset['train_set_y'][:])

    labels = labels.reshape((1, labels.shape[0]))

    global num_px
    num_px = data[0].shape[0]

    return data, labels


def process_image(fname, weigths, bias):
    fname = 'images/' + fname
    img = np.array(ndimage.imread(fname, flatten=False))
    res = misc.imresize(
        img, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T

    prediction, activation = algorithms.predict(weigths, bias, res)

    return prediction, activation


def dump_model(weights, bias):
    np.savetxt('weights', weights)
    np.savetxt('bias', bias)


def load_model():
    return np.load('weights'), np.load('bias')


def main():
    data, labels = load_data()

    # Dataset standardization
    flatten_data = data.reshape(data.shape[0], -1).T
    training_set = flatten_data / 255.0

    # Train
    costs, prediction, weights, bias, lrate, niter = algorithms.model(
        training_set, labels, niter=5000, lrate=0.01, verbose=True)

    # 2000,  .005, 99.04306220095694, 0.14087207570310153
    # 5000,  .01,  100.0,             0.03149755317976998
    # 1000,  .01,  98.56459330143541, 0.12497148000976802

    dump_model(weights, bias)

    print(process_image('r1.jpg', weights, bias))
    print(process_image('r2.jpg', weights, bias))
    print(process_image('r3.jpg', weights, bias))
    print(process_image('r4.jpg', weights, bias))


if __name__ == '__main__':
    main()