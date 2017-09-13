import argparse
import h5py
import algorithms

import numpy as np
from scipy import ndimage, misc

num_px = 64


def load_data():
    dataset = h5py.File('./data/train_catvnoncat.h5', 'r')

    data = np.array(dataset['train_set_x'][:])
    labels = np.array(dataset['train_set_y'][:])

    labels = labels.reshape((1, labels.shape[0]))

    return data, labels


def process_image(fname, weigths, bias):
    img = np.array(ndimage.imread(fname, flatten=False))
    res = misc.imresize(
        img, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T

    prediction = algorithms.predict(weigths, bias, res)

    return prediction


def dump_model(weights, bias, verbose=False):
    np.savetxt('weights', weights)
    if verbose: print(f'SAVED: weights {weights.shape}')

    np.savetxt('bias', (bias, ))
    if verbose: print(f'SAVED: bias {bias}')


def load_model(verbose=False):
    weights = np.loadtxt('weights')
    if verbose: print(f'LOADED: weights {weights.shape}')

    bias = np.loadtxt('bias')
    if verbose: print(f'LOADED: bias {bias}')

    return weights, bias


def main(args):
    verbose = True

    if args.retrain:
        if verbose: print('retraining model')
        data, labels = load_data()

        # Dataset standardization
        flatten_data = data.reshape(data.shape[0], -1).T
        training_set = flatten_data / 255.0

        # Train
        niter = 1000
        lrate = 0.005

        costs, prediction, weights, bias, lrate, niter = algorithms.model(
            training_set, labels, niter, lrate, verbose)

        # Dumb a new model
        dump_model(weights, bias, verbose)

    
    if args.dump:
        if verbose: print('loading model')
        weights, bias = load_model(verbose)

    if args.input:
        print(process_image(args.input, weights, bias))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cat or not neural network')
    parser.add_argument('-d', '--dump', action='store_true', help='use model dump')
    parser.add_argument('-r', '--retrain', action='store_true', help='retrain model')
    parser.add_argument('-i', '--input', help='path to input image')
    
    args = parser.parse_args()

    main(args)