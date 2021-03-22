import keras
import warnings
import keras.backend as K
import numpy as np


class WarmUpReduceLROnPlateauScheduler(keras.callbacks.Callback):
    """
    ReduceLROnPlateau + WarmUp Scheduler
    =========================================================
    Reduce learning rate when a metric has stopped improving.
        Models often benefit from reducing the learning rate by a factor
        of 2-10 once learning stagnates. This callback monitors a
        quantity and if no improvement is seen for a 'patience' number
        of epochs, the learning rate is reduced.
        # Example
        ```python
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.001)
        model.fit(X_train, Y_train, callbacks=[reduce_lr])
        ```
        # Arguments
            monitor: quantity to be monitored.
            factor: factor by which the learning rate will
                be reduced. new_lr = lr * factor
            patience: number of epochs that produced the monitored
                quantity with no improvement after which training will
                be stopped.
                Validation quantities may not be produced for every
                epoch, if the validation frequency
                (`model.fit(validation_freq=5)`) is greater than one.
            verbose: int. 0: quiet, 1: update messages.
            mode: one of {auto, min, max}. In `min` mode,
                lr will be reduced when the quantity
                monitored has stopped decreasing; in `max`
                mode it will be reduced when the quantity
                monitored has stopped increasing; in `auto`
                mode, the direction is automatically inferred
                from the name of the monitored quantity.
            min_delta: threshold for measuring the new optimum,
                to only focus on significant changes.
            cooldown: number of epochs to wait before resuming
                normal operation after lr has been reduced.
            min_lr: lower bound on the learning rate.
        """

    def __init__(self, obj_lr, warmup_steps, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0,
                 **kwargs):
        super(WarmUpReduceLROnPlateauScheduler, self).__init__()
        self.warmup_steps = warmup_steps
        self.cur_steps = 0
        self.obj_lr = obj_lr
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('`epsilon` argument is deprecated and '
                          'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        if self.cur_steps < self.warmup_steps:
            new_lr = self.cur_steps / self.warmup_steps * self.obj_lr
            self.cur_steps += 1
            K.set_value(self.model.optimizer.lr, new_lr)
            logs['lr'] = K.get_value(self.model.optimizer.lr)
            if self.verbose > 1:
                print('\niter %05d: Warmup learning rate to %s.' % (self.cur_steps + 1, new_lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing '
                                  'learning rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


class WarmUpScheduler(keras.callbacks.Callback):
    """
    ReduceLROnPlateau + WarmUp Scheduler
    =========================================================
    Reduce learning rate when a metric has stopped improving.
        Models often benefit from reducing the learning rate by a factor
        of 2-10 once learning stagnates. This callback monitors a
        quantity and if no improvement is seen for a 'patience' number
        of epochs, the learning rate is reduced.
        # Example
        ```python
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.001)
        model.fit(X_train, Y_train, callbacks=[reduce_lr])
        ```
        # Arguments
            monitor: quantity to be monitored.
            factor: factor by which the learning rate will
                be reduced. new_lr = lr * factor
            patience: number of epochs that produced the monitored
                quantity with no improvement after which training will
                be stopped.
                Validation quantities may not be produced for every
                epoch, if the validation frequency
                (`model.fit(validation_freq=5)`) is greater than one.
            verbose: int. 0: quiet, 1: update messages.
            mode: one of {auto, min, max}. In `min` mode,
                lr will be reduced when the quantity
                monitored has stopped decreasing; in `max`
                mode it will be reduced when the quantity
                monitored has stopped increasing; in `auto`
                mode, the direction is automatically inferred
                from the name of the monitored quantity.
            min_delta: threshold for measuring the new optimum,
                to only focus on significant changes.
            cooldown: number of epochs to wait before resuming
                normal operation after lr has been reduced.
            min_lr: lower bound on the learning rate.
        """

    def __init__(self, obj_lr, warmup_steps, verbose=0):
        super(WarmUpScheduler, self).__init__()
        self.warmup_steps = warmup_steps
        self.cur_steps = 0
        self.obj_lr = obj_lr
        self.verbose = verbose

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        if self.cur_steps < self.warmup_steps:
            new_lr = self.cur_steps / self.warmup_steps * self.obj_lr
            self.cur_steps += 1
            K.set_value(self.model.optimizer.lr, new_lr)
            logs['lr'] = K.get_value(self.model.optimizer.lr)
            if self.verbose > 1:
                print('\niter %05d: Warmup learning rate to %s.' % (self.cur_steps + 1, new_lr))
