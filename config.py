"""
Config module for tensorflow speech recognition commands.

created on 30.08.2021

@author: Ruslan Dolovaniuk

"""

import os


class Config:
    """Config class for project."""

    def __init__(self):
        self.seed = 42
        self.data_dir = 'data'

        self.dataset_dir = 'mini_speech_commands'
        self.dataset_file = 'mini_speech_commands.zip'
        self.dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip'

        self.total = 0
        self.__train = 80
        self.__val = 10
        self.__test = 10

        self.samples = 16000
        self.frame_length = 255
        self.frame_step = 128

        self.batch_size = 64
        self.resize_size = 32
        self.kernel_size = 3
        self.dropout = 0.25
        self.EPOCHS = 10

    @property
    def dataset_path(self):
        return os.path.join(self.data_dir, self.dataset_dir)

    @property
    def train(self):
        return int((self.total*self.__train)/100)

    @property
    def val(self):
        return int((self.total*self.__val)/100)

    @property
    def test(self):
        return int((self.total*self.__test)/100)
