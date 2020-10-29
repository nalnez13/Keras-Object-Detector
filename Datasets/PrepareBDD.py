import json
import os
import cv2
import tqdm
import shutil


def parse_json(json_file, data_type):
    if not os.path.isdir(dst_dir + '/day'):
        os.mkdir(dst_dir + '/day')
    if not os.path.isdir(dst_dir + '/' + 'day/' + data_type):
        os.mkdir(dst_dir + '/' + 'day/' + data_type)
    if not os.path.isdir(dst_dir + '/' + 'day/' + data_type + '/image'):
        os.mkdir(dst_dir + '/' + 'day/' + data_type + '/image')
    if not os.path.isdir(dst_dir + '/' + 'day/' + data_type + '/annotation'):
        os.mkdir(dst_dir + '/' + 'day/' + data_type + '/annotation')

    if not os.path.isdir(dst_dir + '/night'):
        os.mkdir(dst_dir + '/night')
    if not os.path.isdir(dst_dir + '/' + 'night/' + data_type):
        os.mkdir(dst_dir + '/' + 'night/' + data_type)
    if not os.path.isdir(dst_dir + '/' + 'night/' + data_type + '/image'):
        os.mkdir(dst_dir + '/' + 'night/' + data_type + '/image')
    if not os.path.isdir(dst_dir + '/' + 'night/' + data_type + '/annotation'):
        os.mkdir(dst_dir + '/' + 'night/' + data_type + '/annotation')

    with open(json_file, 'r') as f:
        read_data = json.load(f)
        for img_data in tqdm.tqdm(read_data):
            img_name = img_data['name']
            time_data = img_data['attributes']['timeofday']

            if time_data == 'night':
                save_path = dst_dir + '/night/' + data_type
            else:
                save_path = dst_dir + '/day/' + data_type
            write_data = open(save_path + '/annotation/' + img_name.replace('.jpg', '') + '.txt', 'w')
            shutil.copy(data_dir + 'images/100k/' + data_type + '/' + img_name, save_path + '/image')
            img = cv2.imread(data_dir + 'images/100k/' + data_type + '/' + img_name)
            im_h, im_w, c = img.shape
            # if time_data not in ['daytime', 'night', 'dawn/dusk']:
            #     print(time_data)
            #     cv2.imshow('img', img)
            #     cv2.waitKey(0)
            labels = img_data['labels']
            for label_data in labels:
                category = label_data['category']
                if category in dst_classes:
                    cid = dst_classes.index(category)
                    x1 = label_data['box2d']['x1']
                    y1 = label_data['box2d']['y1']
                    x2 = label_data['box2d']['x2']
                    y2 = label_data['box2d']['y2']
                    area = (x2 - x1) * (y2 - y1)
                    if area < 900:
                        continue

                    img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cx = ((x1 + x2) / 2.) / im_w
                    cy = ((y1 + y2) / 2.) / im_h
                    w = (x2 - x1) / im_w
                    h = (y2 - y1) / im_h

                    write_str = ' '.join(list(map(str, [cid, cx, cy, w, h]))) + '\n'
                    write_data.write(write_str)
            write_data.close()
            # cv2.imshow('img', img)
            # cv2.waitKey(1)


data_dir = 'E:/bdd100k/'
dst_classes = ['car', 'bus', 'truck']
dst_dir = 'E:/bdd100k/time_parsed'

if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)

# valid set
parse_json(data_dir + 'labels/bdd100k_labels_images_val.json', 'val')
# train set
parse_json(data_dir + 'labels/bdd100k_labels_images_train.json', 'train')
