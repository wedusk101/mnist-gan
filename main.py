#!/usr/bin/env python
#
import argparse
import tensorflow as tf
import numpy as np
import os
import sys
import code
import random
from gan import GAN
from mnist import MNIST

def gen_samples(gan, sessions): 
    samples = []
    for i, s in enumerate(sessions):
        samples_for_digit = gan.eval_generator(s, 32)
        for sample in samples_for_digit:
            samples.append((sample, i))
    random.shuffle(samples)
    samples = zip(*samples)
    samples[0] = np.asarray(samples[0])
    samples[1] = tf.contrib.learn.python.learn.datasets.mnist.dense_to_one_hot(np.asarray(samples[1]), 10)
    return samples
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-dir", default='/tmp/mnist-data', help="Directory where mnist downloaded dataset will be stored")
    parser.add_argument("--output-dir", default='output', help="Directory where models will be saved")
    parser.add_argument("--train-digits", help="Comma separated list of digits to train generators for (e.g. '1,2,3')")
    parser.add_argument("--train-mnist", action='store_true', help="If specified, train the mnist classifier based on generated digits from saved models")
    global args
    
    args = parser.parse_args() # used to store input arguments    
    
    mnist_data = tf.contrib.learn.datasets.mnist.read_data_sets(args.mnist_dir, one_hot=True) # loads mnist data from tensorflow datasets

    if args.train_digits:	# checks if the user has input digits to train on
        gan = GAN()
        for digit in map(int, args.train_digits.split(',')): # iterates through a list of user input digits ---- map() applies a function to a list and 
            path = "%s/digit-%d/model" % (args.output_dir, digit) # creates a variable to store the path for saving the models
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            gan.train_digit(mnist_data, digit, path) # reads the mnist data for each digit in the training set provided by the user and saves the trained session to path
    elif args.train_mnist:	# if the user doesn't input any training data
        gan = GAN()
        print("Loading generator models...")
        sessions = [gan.restore_session("%s/digit-%d" % (args.output_dir, digit)) for digit in range(10)] # restores saved generator sessions for each digits 0 through 9
        print("Done")
        samples = [[], []]
        
        mnist = MNIST()
        for step in range(20000):
            if len(samples[0]) < 50:
                samples = gen_samples(gan, sessions)
            xs = samples[0][:50]
            ys = samples[1][:50]
            samples[0] = samples[0][50:]
            samples[1] = samples[1][50:]
            mnist.train_batch(xs, ys, step)
        test_accuracy = mnist.eval_batch(mnist_data.test.images, mnist_data.test.labels)
        print("Test accuracy %g" % test_accuracy)

if __name__ == "__main__":
    main()