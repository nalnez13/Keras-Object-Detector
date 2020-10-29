import glob
from tqdm import tqdm
import random
import os

valid = 0
is_txt = 0
file_path = glob.glob("E:/dataset/****/***/**/*.jpg")

num = int(len(file_path) * 0.1)
random.shuffle(file_path)
with open("valid.txt", "w") as f:
    for j in file_path[0:num]:
        is_txt = 0
        txt_root = j.replace(".jpg", ".txt")
        if os.path.isfile(txt_root):
            f.write(j + "\n")
            valid += 1
print(valid)

with open("train.txt", "w") as f:
    for j in file_path[num::]:
        is_txt = 0
        txt_root = j.replace(".jpg", ".txt")
        if os.path.isfile(txt_root):
            f.write(j + "\n")
            valid += 1
print(valid)
