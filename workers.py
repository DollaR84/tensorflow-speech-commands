"""
Workers module for tensorflow speech recognition commands.

created on 30.08.2021

@author: Ruslan Dolovaniuk

"""

import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

import tools


class Base:
    """Base worker for all workers."""

    def __init__(self, files, commands, cfg):
        self.commands = commands
        self.cfg = cfg
        self.__AUTOTUNE__ = tf.data.AUTOTUNE
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        waveform_ds = files_ds.map(tools.get_waveform_and_label, num_parallel_calls=self.__AUTOTUNE__)
        self.__spectrogram_ds = waveform_ds.map(lambda audio, label: tools.get_spectrogram_and_label_id(audio, label, self.commands, self.cfg), num_parallel_calls=self.__AUTOTUNE__)

    @property
    def output(self):
        return self.__spectrogram_ds


class Trainer(Base):
    """Train worker for wav from train files dataset."""

    def __init__(self, files, commands, cfg):
        super().__init__(files, commands, cfg)

    def preprocessing_and_normalization(self):
        for spectrogram, _ in self.output.take(1):
            input_shape = spectrogram.shape
        norm_layer = preprocessing.Normalization()
        norm_layer.adapt(self.output.map(lambda x, _: x))

        self.model = models.Sequential([
          layers.Input(shape=input_shape, batch_size=self.cfg.batch_size),
          preprocessing.Resizing(self.cfg.resize_size, self.cfg.resize_size),
          norm_layer,
          layers.Conv2D(self.cfg.resize_size, self.cfg.kernel_size, activation='relu'),
          layers.Conv2D(self.cfg.resize_size*2, self.cfg.kernel_size, activation='relu'),
          layers.MaxPooling2D(),
          layers.Dropout(self.cfg.dropout),
          layers.Flatten(),
          layers.Dense(self.cfg.frame_step, activation='relu'),
          layers.Dropout(self.cfg.dropout*2),
          layers.Dense(len(self.commands)),])
        self.model.summary()

    def run(self, val_ds):
        self.model.compile(
          optimizer=tf.keras.optimizers.Adam(),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'],)
        self.history = self.model.fit(
          self.output,
          batch_size=self.cfg.batch_size,
          validation_data=val_ds,
          validation_batch_size=self.cfg.batch_size,
          epochs=self.cfg.EPOCHS,
          callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),)


class Validator(Base):
    """Validation worker for wav from val files dataset."""

    def __init__(self, files, commands, cfg):
        super().__init__(files, commands, cfg)


class Tester(Base):
    """Test worker for wav from test files dataset."""

    def __init__(self, files, commands, cfg):
        super().__init__(files, commands, cfg)

    def run(self, model):
        audios = []
        labels = []
        for audio, label in self.output:
            audios.append(audio.numpy())
            labels.append(label.numpy())

        audios = np.array(audios)
        labels = np.array(labels)

        y_pred = np.argmax(model.predict(audios), axis=1)
        y_true = labels
        self.acc = sum(y_pred == y_true) / len(y_true)
