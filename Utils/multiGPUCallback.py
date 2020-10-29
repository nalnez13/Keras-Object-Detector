from keras.callbacks import ModelCheckpoint
import numpy as np
import warnings


class MultiGPUModelCheckpoint(ModelCheckpoint):

    def __init__(self, model, *args, **kwargs):
        super(MultiGPUModelCheckpoint, self).__init__(*args, **kwargs)
        self.model = model

    def set_model(self, model):
        pass

