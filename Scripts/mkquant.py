import cv2
import glob
import random
import os

if not os.path.isdir('./quantizeset'):
    os.mkdir('./quantizeset')

# imgs = glob.glob('E:/bdd100k/time_parsed/day/val/image/*.jpg')
with open('train.txt', 'r') as f:
    imgs = f.read().splitlines()
random.shuffle(imgs)

with open('quantize_list.txt', 'w') as f:
    for i in range(10000):
        im = cv2.imread(imgs[i])
        im = cv2.resize(im, (224, 224))
        cv2.imwrite('quantizeset/{}.jpg'.format(i), im)
        f.write('quantizeset/{}.jpg\n'.format(i))
