import keras
from keras_radam import RAdam
import Generator
from models import Head, Backbones, LossFunc
from default_config import cluster_configs
from imgaug import augmenters as iaa
import datetime
import os
import tensorflow as tf
import argparse
import tqdm
from Utils.print_color import bcolors
from Utils.cyclical_learning_rate import CyclicLR
from Utils.multiGPUCallback import MultiGPUModelCheckpoint
from Utils.backboneCheckPoint import BackboneCheckPoint

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)


def get_single_gpu_model(input_size, num_classes, anchors_per_layer, weight_decay):
    backbone, s1, s2, s3, s4, s5, s6 = Backbones.GMB_CPRN(input_shape=(input_size, input_size, 3),
                                                          weight_decay=weight_decay)
    model = Head.D_FPN_subnet_fix(backbone, s3, s4, s6, num_classes, anchors_per_layer, weight_decay=weight_decay)
    return backbone, model


def get_multi_gpu_model(input_size, num_classes, anchors_per_layer, gpus, weight_decay):
    with tf.device("/cpu:0"):
        backbone, s1, s2, s3, s4, s5, s6 = Backbones.GMB_CPRN(input_shape=(input_size, input_size, 3),
                                                              weight_decay=weight_decay)
        model = Head.D_FPN_subnet_fix(backbone, s3, s4, s6, num_classes, anchors_per_layer, weight_decay=weight_decay)
    multi_gpu_model = keras.utils.multi_gpu_model(model, gpus=gpus)

    return backbone, model, multi_gpu_model


def Train(num_classes, train, val, name, epochs=5000, workers=4, gpus=1,
          load_backbone=True, input_size=256, freeze_backbone=False, batch_size=32, lr=0.01, weights=None,
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
    # TODO: Multi processing check
    print('[i] weights :', weights)
    print('[i] classes :', num_classes)
    print('[i] input size :', input_size)
    print('[i] train data :', train)
    print('[i] valid data :', val)
    print('[i] gpus: ', gpus)
    print('[i] epochs:', epochs)
    print('[i] load only backbone: ', load_backbone and weights is not None)
    print('[i] freeze backbone:', freeze_backbone)
    print('[i] batch size:', batch_size)
    print('[i] learning rate:', lr)
    print('[i] weight_decay:', weight_decay)
    print('=======================================' + bcolors.ENDC)

    augments = [iaa.SomeOf((0, 3),
                           [
                               iaa.Identity(),
                               iaa.Rotate((-5, 5)),
                               iaa.Sharpen(),
                               iaa.TranslateX(),
                               iaa.GammaContrast(),
                               iaa.ShearX((-5, 5)),
                               iaa.TranslateY(),
                               iaa.MultiplyHueAndSaturation(mul=(0.8, 1.5)),
                               iaa.MultiplyAndAddToBrightness(),
                               iaa.ShearY((-5, 5))
                           ]),
                iaa.Fliplr(p=0.5),
                iaa.Sometimes(0.3, iaa.Grayscale(alpha=(0.1, 1.0))),
                iaa.Sometimes(0.2, iaa.Dropout(p=(0.01, 0.1))),
                iaa.Sometimes(0.3, iaa.GammaContrast(gamma=(0.3, 1.2))),  # Gamma Distortion
                iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.0, 1.0))),  # Gaussian Blur
                iaa.SomeOf((0, 3),
                           [iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.2, 0.2), keep_size=True)),
                            iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                                          order=[0, 1])),
                            iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.05, 0.2)))])
                ]

    train_batch_gen = Generator.AnchorGenerator(batch_size=batch_size, input_shape=(input_size, input_size, 3),
                                                num_classes=num_classes, anchor_config=cluster_configs, data_path=train,
                                                augs=augments, is_train=True)

    valid_batch_gen = Generator.AnchorGenerator(batch_size=batch_size, input_shape=(input_size, input_size, 3),
                                                num_classes=num_classes, anchor_config=cluster_configs, data_path=val,
                                                augs=[], is_train=False)

    callbacks = [
        CyclicLR(max_lr=lr, base_lr=1e-7, step_size=5000),
        keras.callbacks.TensorBoard(log_dir),
    ]

    if gpus > 1:
        backbone, cpu_model, model = get_multi_gpu_model(input_size, num_classes, 3, gpus, weight_decay)
        callbacks.append(MultiGPUModelCheckpoint(model=cpu_model, filepath='./saved_models/' + name + '-{epoch:05d}.h5',
                                                 verbose=1, period=5, save_best_only=True))
    else:
        backbone, model = get_single_gpu_model(input_size, num_classes, 3, weight_decay)
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

    model.compile(keras.optimizers.SGD(lr, momentum=0.9, nesterov=True),
                  loss=LossFunc.Multibox_Loss(cluster_configs).compute_loss)

    model.fit_generator(train_batch_gen,
                        use_multiprocessing=True,
                        max_queue_size=10,
                        callbacks=callbacks,
                        workers=workers,
                        epochs=epochs,
                        validation_data=valid_batch_gen, validation_freq=5)

    model.save('./saved_models/final.h5')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--classes', required=True, type=int, help='# of classes in Dataset')
    # parser.add_argument('--name', required=True, type=str, help='Set model name for log, saving weights')
    # parser.add_argument('--train', required=True, type=str, help='Training dataset path or list text file')
    # parser.add_argument('--val', required=True, type=str, help='Validation dataset path or list text file')
    #
    # parser.add_argument('--epochs', required=False, type=int, default=5000, help='# of training epochs')
    # parser.add_argument('--workers', required=False, type=int, default=4, help='# of multi processing workers')
    # parser.add_argument('--gpus', required=False, type=int, default=1, help='# of gpus for training')
    # parser.add_argument('--load_backbone', required=False, action='store_true',
    #                     help='Load only weights of backbone layers')
    # parser.add_argument('--freeze_backbone', required=False, action='store_true', help='Freeze backbone layers')
    # parser.add_argument('--batch_size', required=False, type=int, default=32, help='# of Training batch size')
    # parser.add_argument('--input_size', required=True, type=int, default=256, help='Size of Input Size')
    # parser.add_argument('--lr', required=False, type=float, default=0.01)
    # parser.add_argument('--weights', required=False, type=str, default=None, help='Weights file to load')
    # parser.add_argument('--weight_decay', required=False, type=float, default=0.0001, help='Weights Decay')
    #
    # args = parser.parse_args()
    # Train(num_classes=args.classes, train=args.train, val=args.val, name=args.name, epochs=args.epochs,
    #       workers=args.workers, gpus=args.gpus, load_backbone=args.load_backbone, freeze_backbone=args.freeze_backbone,
    #       batch_size=args.batch_size, lr=args.lr, weights=args.weights, weight_decay=args.weight_decay)
    
    Train(num_classes=3, train='train.txt', val='valid.txt',
          name='day_ContextModule', input_size=256, lr=0.01, gpus=1, batch_size=64)
