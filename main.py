"""
Main module for run tensorflow speech recognition commands.

created on 30.08.2021

@author: Ruslan Dolovaniuk

"""

import os

import numpy as np

#import seaborn as sns

import tensorflow as tf

from config import Config

from workers import Trainer, Validator, Tester


class Initializer:
    """Init variables after start project."""

    def __init__(self, config):
        self.config = config
        tf.random.set_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def dataset(self):
        if not os.path.exists(self.config.data_dir):
            os.mkdir(self.config.data_dir)
        if not os.path.exists(self.config.dataset_path):
            tf.keras.utils.get_file(
              self.config.dataset_file,
              origin=self.config.dataset_url,
              extract=True,
              cache_dir='.', cache_subdir=self.config.data_dir)

    def get_commands(self):
        commands = np.array(tf.io.gfile.listdir(self.config.dataset_path))
        commands = commands[commands != 'README.md']
        return commands

    def get_files(self, commands):
        filenames = tf.io.gfile.glob(self.config.dataset_path + '/*/*')
        filenames = tf.random.shuffle(filenames)
        num_samples = len(filenames)
        self.config.total = num_samples
        return filenames

    def get_train_files(self, files):
        return files[:self.config.train]

    def get_val_files(self, files):
        return files[self.config.train:self.config.train+self.config.val]

    def get_test_files(self, files):
        return files[-self.config.test:]


def main():
    cfg = Config()
    init = Initializer(cfg)
    init.dataset()
    commands = init.get_commands()
    files = init.get_files(commands)
    train_files = init.get_train_files(files)
    val_files = init.get_val_files(files)
    test_files = init.get_test_files(files)

    print('Commands:', commands)
    print('Number of total examples:', cfg.total)
    print('Number of examples per label:', len(tf.io.gfile.listdir(os.path.join(cfg.dataset_path, commands[0]))))
    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))

    train = Trainer(train_files, commands, cfg)
    val = Validator(val_files, commands, cfg)
    test = Tester(test_files, commands, cfg)

    train.preprocessing_and_normalization()
    train.run(val.output)
    test.run(train.model)
    print(f'Test set accuracy: {test.acc:.0%}')


if '__main__' == __name__:
    main()
