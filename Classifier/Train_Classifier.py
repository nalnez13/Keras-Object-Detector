import tensorflow as tf
import keras
import datetime
import os
import argparse
from swa.keras import SWA
from Classifier import Generator
from models import Backbones
from imgaug import augmenters as iaa
from Utils.print_color import bcolors
from Utils.cyclical_learning_rate import CyclicLR
from Utils.warmup_scheduler import WarmUpScheduler
from Utils.multiGPUCallback import MultiGPUModelCheckpoint

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)


def get_single_gpu_model(input_size, num_classes, weight_decay):
    model, s1, s2, s3, s4, s5, s6 = Backbones.GhostNet_CRELU_CSP(input_shape=(input_size, input_size, 3),
                                                                 weight_decay=weight_decay, num_classes=num_classes)
    return model


def get_multi_gpu_model(input_size, num_classes, gpus, weight_decay):
    with tf.device("/cpu:0"):
        model, s1, s2, s3, s4, s5, s6 = Backbones.v4_tiny(input_shape=(input_size, input_size, 3),
                                                          weight_decay=weight_decay, num_classes=num_classes)
    multi_gpu_model = keras.utils.multi_gpu_model(model, gpus=gpus)

    return model, multi_gpu_model


def Train(name, epochs=5000, workers=4, gpus=1, input_size=64, batch_size=32, lr=0.01, weights=None,
          weight_decay=0.0001):
    if not os.path.isdir('./saved_models'):
        os.makedirs('./saved_models')
    if not os.path.isdir('./logs'):
        os.makedirs('./logs')

    log_dir = os.path.join(
        "logs", "fit", name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), )

    # Training Setup
    print(bcolors.GREEN)
    print('=======================================')
    print('[i] name :', name)
    print('[i] input size :', input_size)
    print('[i] epochs:', epochs)
    print('[i] batch size:', batch_size)
    print('[i] learning rate:', lr)
    print('[i] gpus: ', gpus)
    print('[i] weights :', weights)
    print('[i] weight_decay:', weight_decay)
    print('=======================================' + bcolors.ENDC)

    augments = [iaa.SomeOf((0, 7),
                           [
                               iaa.Identity(),
                               iaa.Rotate(),
                               iaa.Posterize(),
                               iaa.Sharpen(),
                               iaa.TranslateX(),
                               iaa.GammaContrast(),
                               iaa.Solarize(),
                               iaa.ShearX(),
                               iaa.TranslateY(),
                               iaa.HistogramEqualization(),
                               iaa.MultiplyHueAndSaturation(),
                               iaa.MultiplyAndAddToBrightness(),
                               iaa.ShearY(),
                               iaa.ScaleX(),
                               iaa.ScaleY(),
                               iaa.Fliplr(),
                               iaa.Crop()
                           ])
                ]

    train_batch_gen = Generator.Tiny_imagenet_Generator(batch_size, (64, 64, 3), '../Datasets/tiny-imagenet-200',
                                                        augments,
                                                        is_train=True, label_smoothing=True)

    valid_batch_gen = Generator.Tiny_imagenet_Generator(batch_size, (64, 64, 3), '../Datasets/tiny-imagenet-200',
                                                        [],
                                                        is_train=False, label_smoothing=True)

    callbacks = [
        CyclicLR(max_lr=lr, base_lr=1e-8, step_size=train_batch_gen.__len__() * 10, mode='triangular2'),
        SWA(start_epoch=50, batch_size=batch_size, verbose=1),
        keras.callbacks.TensorBoard(log_dir),
    ]

    if gpus > 1:
        cpu_model, model = get_multi_gpu_model(input_size, 200, gpus, weight_decay)
        callbacks.append(MultiGPUModelCheckpoint(model=cpu_model, filepath='./saved_models/' + name + '-{epoch:05d}.h5',
                                                 verbose=1, period=5, save_best_only=True))
    else:
        model = get_single_gpu_model(input_size, 200, weight_decay)
        callbacks.append(keras.callbacks.ModelCheckpoint(filepath='./saved_models/' + name + '-{epoch:05d}.h5',
                                                         verbose=1, period=5, save_best_only=True))

    if weights:
        model.load_weights(weights)

    model.compile(keras.optimizers.SGD(lr, momentum=0.9, nesterov=True),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy, keras.metrics.top_k_categorical_accuracy])

    model.fit_generator(train_batch_gen,
                        use_multiprocessing=True,
                        max_queue_size=10,
                        callbacks=callbacks,
                        workers=workers,
                        epochs=epochs,
                        validation_data=valid_batch_gen, validation_freq=2)

    model.save('./saved_models/final.h5')


if __name__ == '__main__':
    use_args = False
    if use_args:
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', required=True, type=str, help='Set model name for log, saving weights')
        parser.add_argument('--epochs', required=False, type=int, default=5000, help='# of training epochs')
        parser.add_argument('--workers', required=False, type=int, default=4, help='# of multi processing workers')
        parser.add_argument('--gpus', required=False, type=int, default=1, help='# of gpus for training')
        parser.add_argument('--batch_size', required=False, type=int, default=32, help='# of Training batch size')
        parser.add_argument('--lr', required=False, type=float, default=0.01)
        parser.add_argument('--weights', required=False, type=str, default=None, help='Weights file to load')
        parser.add_argument('--weight_decay', required=False, type=float, default=0.0001, help='Weights Decay')

        args = parser.parse_args()
        Train(name=args.name, epochs=args.epochs,
              workers=args.workers, gpus=args.gpus,
              batch_size=args.batch_size, lr=args.lr, weights=args.weights, weight_decay=args.weight_decay)

    else:
        Train(name='t-img-GhostNet_CRelu_CSP', input_size=64, lr=0.1, gpus=1, batch_size=256, epochs=300,
              weight_decay=0.0005)
