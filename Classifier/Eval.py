from models import Backbones
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
import cv2
from natsort import natsorted
import os
import tensorflow as tf
import keras

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)
keras.backend.set_learning_phase(0)

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    num_class = 1000
    with tf.device("/cpu:0"):
        model, s1, s2, s3, s4, s5, s6 = Backbones.GhostNet_CRELU_CSP_Large(input_shape, num_class)
    model.load_weights('saved_models/LP-GhostNet_CRelu_CSP_Large-00100.h5')
    test_imgs = natsorted(glob("../Datasets/LPD_competition/test/**/*.jpg", recursive=True))
    columns = ["filename", "prediction"]
    filename_list = []
    prediction_list = []
    for img_path in tqdm(test_imgs):
        img_path = img_path.replace("\\", "/")
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=0)
        result = model.predict(img)
        cls = np.argmax(result)
        filename_list.append(filename)
        prediction_list.append(cls)

    dataframe = pd.DataFrame({'filename': filename_list, 'prediction': prediction_list})
    dataframe.to_csv("inference_result.csv", index=False)
