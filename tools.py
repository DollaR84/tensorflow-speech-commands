"""
Tools module for tensorflow speech recognition commands.

created on 30.08.2021

@author: Ruslan Dolovaniuk

"""

import os

import tensorflow as tf


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform, cfg):
    zero_padding = tf.zeros([cfg.samples] - tf.shape(waveform), dtype=tf.float32)

    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
      equal_length,
      frame_length=cfg.frame_length,
      frame_step=cfg.frame_step)
    spectrogram = tf.abs(spectrogram)
    return spectrogram


def get_spectrogram_and_label_id(audio, label, commands, cfg):
    spectrogram = get_spectrogram(audio, cfg)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id
