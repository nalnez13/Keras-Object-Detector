import glob
import os

imgs = glob.glob('//192.168.0.7/sangmin/hh/dataset/****/***/**/*.jpg')

for i in imgs:
    txt = i.replace('.jpg', '.txt')
    if os.path.isfile(txt):
        with open(txt, 'r') as f:
            annotations = f.read().splitlines()
            for annot in annotations:
                annot = annot.split(' ')
                class_id = int(annot[0])
                cx = float(annot[1])
                cy = float(annot[2])
                w = float(annot[3])
                h = float(annot[4])
                if w < 0.001 or h < 0.001:
                    print(txt)
