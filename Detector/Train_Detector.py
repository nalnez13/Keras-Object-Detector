import tensorflow as tf
import keras
import datetime
import os
import argparse
from swa.keras import SWA
from keras_radam import RAdam
from Detector import Generator
from Detector import AnchorPresets, AnchorUtil
from models import Head, Backbones, LossFunc
from imgaug import augmenters as iaa
import imgaug
from Utils.print_color import bcolors
from Utils.cyclical_learning_rate import CyclicLR
from Utils.multiGPUCallback import MultiGPUModelCheckpoint
from Utils.backboneCheckPoint import BackboneCheckPoint

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)


def get_single_gpu_model(num_classes, anchors_per_layer, weight_decay):
    backbone, s1, s2, s3, s4, s5, s6 = Backbones.GhostNet_CRELU_CSP(input_shape=(None, None, 3),
                                                                    weight_decay=weight_decay)
    model = Head.SharedHeadNet(backbone, s3, s4, s5, s6, num_classes, anchors_per_layer, weight_decay=weight_decay)
    return backbone, model


def get_multi_gpu_model(input_size, num_classes, anchors_per_layer, gpus, weight_decay):
    with tf.device("/cpu:0"):
        backbone, s1, s2, s3, s4, s5, s6 = Backbones.v4_tiny(input_shape=(input_size, input_size, 3),
                                                             weight_decay=weight_decay)
        model = Head.SubNet(backbone, s4, s6, num_classes, anchors_per_layer, weight_decay=weight_decay)
    multi_gpu_model = keras.utils.multi_gpu_model(model, gpus=gpus)

    return backbone, model, multi_gpu_model


def Train(num_classes, train, val, name,
          epochs=5000, workers=4, gpus=1,
          batch_size=32, base_input_size=256, lr=0.01,
          weights=None, freeze_backbone=False, load_backbone=False,
          multi_scaling=False, multi_scaling_freq=30,
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
    print('[i] base input size :', base_input_size)
    print('[i] classes :', num_classes)
    print('[i] epochs:', epochs)
    print('[i] batch size:', batch_size)
    print('[i] learning rate:', lr)
    print('[i] train data :', train)
    print('[i] valid data :', val)
    print('[i] gpus: ', gpus)
    print('[i] weights :', weights)
    print('[i] load only backbone: ', load_backbone and weights is not None)
    print('[i] freeze backbone:', freeze_backbone)
    print('[i] weight_decay:', weight_decay)
    print('[i] multi scaling:', multi_scaling)
    print('=======================================' + bcolors.ENDC)

    anchor_config = AnchorUtil.AnchorComputer(
        (base_input_size, base_input_size, 3),
        num_classes,
        AnchorPresets.default_config)

    augments = [iaa.SomeOf((0, 3),
                           [
                               iaa.Identity(),
                               iaa.Sharpen(),
                               iaa.GammaContrast(),
                               iaa.MultiplyHueAndSaturation(mul=(0.8, 1.5)),
                               iaa.MultiplyAndAddToBrightness(),
                           ]),
                iaa.Fliplr(p=0.5),
                iaa.Sometimes(0.3, iaa.Grayscale(alpha=(0.1, 1.0))),
                iaa.Sometimes(0.2, iaa.Dropout(p=(0.01, 0.1))),
                iaa.Sometimes(0.3, iaa.GammaContrast(gamma=(0.3, 1.2))),  # Gamma Distortion
                iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.0, 1.0))),  # Gaussian Blur
                iaa.SomeOf((0, 3),
                           [iaa.Sometimes(0.5,
                                          iaa.CropAndPad(percent=(-0.2, 0.2), keep_size=True, pad_mode=imgaug.ALL)),
                            iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                                          mode=imgaug.ALL)),
                            iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.05, 0.05), mode=imgaug.ALL))])
                ]

    train_batch_gen = Generator.AnchorGenerator(batch_size=batch_size,
                                                input_shape=(base_input_size, base_input_size, 3),
                                                num_classes=num_classes, anchor_util=anchor_config, data_path=train,
                                                augs=augments, is_train=True,
                                                multi_scaling=multi_scaling,
                                                scaling_freq=multi_scaling_freq)

    valid_batch_gen = Generator.AnchorGenerator(batch_size=batch_size,
                                                input_shape=(base_input_size, base_input_size, 3),
                                                num_classes=num_classes, anchor_util=anchor_config, data_path=val,
                                                augs=[], is_train=False)

    callbacks = [
        CyclicLR(max_lr=lr, base_lr=1e-7, step_size=train_batch_gen.__len__() * 10, mode='triangular2'),
        SWA(start_epoch=50, batch_size=batch_size, lr_schedule='manual', verbose=1),
        keras.callbacks.TensorBoard(log_dir),
    ]

    if gpus > 1:
        backbone, cpu_model, model = get_multi_gpu_model(num_classes, 6, gpus, weight_decay)
        callbacks.append(MultiGPUModelCheckpoint(model=cpu_model, filepath='./saved_models/' + name + '-{epoch:05d}.h5',
                                                 verbose=1, period=5, save_best_only=True))
    else:
        backbone, model = get_single_gpu_model(num_classes, 6,
                                               weight_decay)
        callbacks.append(keras.callbacks.ModelCheckpoint(filepath='./saved_models/' + name + '-{epoch:05d}.h5',
                                                         verbose=1, period=5, save_best_only=True))

    callbacks.append(BackboneCheckPoint(model=backbone, filepath='./saved_models/' + name + 'backbone-{epoch:05d}.h5',
                                        verbose=1, period=5, save_best_only=True))

    if weights:
        if load_backbone:
            print(bcolors.GREEN + "[i] Load Backbone layers" + bcolors.ENDC)
            backbone.load_weights(weights)
        else:
            model.load_weights(weights)
        if freeze_backbone:
            print(bcolors.YELLOW + "[!] Freeze Backbone Layers" + bcolors.ENDC)
            backbone.trainable = False

    model.compile(RAdam(lr),
                  loss={'prediction': LossFunc.MultiBoxLoss().compute_loss})

    model.fit_generator(train_batch_gen,
                        use_multiprocessing=True,
                        max_queue_size=10,
                        callbacks=callbacks,
                        workers=workers,
                        epochs=epochs,
                        validation_data=valid_batch_gen, validation_freq=5)

    model.save('./saved_models/final.h5')


if __name__ == '__main__':
    use_args = False
    if use_args:
        parser = argparse.ArgumentParser()
        parser.add_argument('--classes', required=True, type=int, help='# of classes in Dataset')
        parser.add_argument('--name', required=True, type=str, help='Set model name for log, saving weights')
        parser.add_argument('--train', required=True, type=str, help='Training dataset path or list text file')
        parser.add_argument('--val', required=True, type=str, help='Validation dataset path or list text file')

        parser.add_argument('--epochs', required=False, type=int, default=5000, help='# of training epochs')
        parser.add_argument('--workers', required=False, type=int, default=4, help='# of multi processing workers')
        parser.add_argument('--gpus', required=False, type=int, default=1, help='# of gpus for training')
        parser.add_argument('--load_backbone', required=False, action='store_true',
                            help='Load only weights of backbone layers')
        parser.add_argument('--freeze_backbone', required=False, action='store_true', help='Freeze backbone layers')
        parser.add_argument('--batch_size', required=False, type=int, default=32, help='# of Training batch size')
        parser.add_argument('--input_size', required=True, type=int, default=256, help='Size of Input Size')
        parser.add_argument('--lr', required=False, type=float, default=0.01)
        parser.add_argument('--weights', required=False, type=str, default=None, help='Weights file to load')
        parser.add_argument('--weight_decay', required=False, type=float, default=0.0001, help='Weights Decay')

        args = parser.parse_args()
        Train(num_classes=args.classes, train=args.train, val=args.val, name=args.name, epochs=args.epochs,
              workers=args.workers, gpus=args.gpus, load_backbone=args.load_backbone,
              freeze_backbone=args.freeze_backbone,
              batch_size=args.batch_size, lr=args.lr, weights=args.weights, weight_decay=args.weight_decay)

    else:
        Train(num_classes=20, train='../Datasets/voc_train.txt', val='../Datasets/voc_valid.txt',
              workers=4,
              name='voc-GhostNet_CRELU_CSP_SharedHead_256_multiscaling', base_input_size=256, lr=0.001, gpus=1,
              batch_size=16, epochs=300, multi_scaling=True, multi_scaling_freq=50)
