import math
from keras.callbacks import Callback
from keras import backend as K


class CosineAnnealingScheduler(Callback):
    def __init__(self, eta_max, eta_min=0, T_max=10, T_mult=2, verbose=0, warmup_steps=0, lr_decay=1.):
        """
        CosineAnnealingScheduler
        :param eta_max: maximum learning rate
        :param eta_min: minimum learning rate
        :param T_max: Cosine duration
        :param T_mult: Consine duration multiplier
        :param verbose: Print Log
        :param warmup_steps: Warm-up Starting Steps
        """
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.prev_epochs = 0
        self.current_steps = 0
        self.warmup_steps = warmup_steps
        self.current_lr = 0
        self.lr_decay = lr_decay
        self.lr_decay_step = 0

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if self.current_steps < self.warmup_steps:
            lr = self.current_lr * (self.current_steps + 1) / self.warmup_steps
            self.current_steps += 1
            K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        e = epoch - self.prev_epochs

        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * e / self.T_max)) / 2
        if lr == self.eta_max:
            self.eta_max *= math.pow(self.lr_decay, self.lr_decay_step)
            lr = self.eta_max
            self.lr_decay_step += 1
            if self.verbose > 0:
                print('\nEpoch %05d: CosineAnnealingScheduler setting maximum eta to %s.' % (epoch + 1, self.eta_max))
        self.current_lr = lr
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))
        if e != 0 and e % self.T_max == 0:
            self.prev_epochs = epoch + 1
            self.T_max *= self.T_mult

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
