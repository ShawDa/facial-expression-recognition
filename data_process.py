# -*- coding:utf-8 -*-

from constants import *
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from os.path import join

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def format_image(image, flag=0):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    if flag == 0:
        gray_border = np.zeros((150, 150), np.uint8)
        gray_border[:, :] = 200
        gray_border[
            int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2)),
            int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2))
        ] = image
        image = gray_border
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=SCALEFACTOR,
        minNeighbors=5
    )

    #  None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
    return image


def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d


def flip_image(image):
    return cv2.flip(image, 1)


def data_to_image(data, i):
    data_image = np.fromstring(
        str(data), dtype=np.uint8, sep=' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()

    # if you want to save all images
    # cv2.imwrite(SAVE_DIRECTORY + '/images/' + str(i) + '.png', data_image)

    data_image = format_image(data_image)
    return data_image


def get_fer(csv_path):
    data = pd.read_csv(csv_path)
    labels = []
    images = []
    count = 0
    total = data.shape[0]

    for index, row in data.iterrows():
        emotion = emotion_to_vec(row['emotion'])
        image = data_to_image(row['pixels'], index)

        if image is not None:
            labels.append(emotion)
            images.append(image)

            # if you want to save faces
            # real_image = image * 255
            # cv2.imwrite(SAVE_DIRECTORY + '/faces/' + str(index) + '.png', real_image)

            count += 1
        print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

    print(index)  # 共35587
    print("Total: " + str(len(images)))
    np.save(join(SAVE_DIRECTORY, 'sf=' + str(SCALEFACTOR) + '_' + SAVE_DATASET_IMAGES_FILENAME), images)
    np.save(join(SAVE_DIRECTORY, 'sf=' + str(SCALEFACTOR) + '_' + SAVE_DATASET_LABELS_FILENAME), labels)


def rafnum2vec(x):
    d = np.zeros(7)
    d[EMOTIONS.index(RAF_EMOTIONS[x-1])] = 1.0
    return d


def image2array(image_path):
    data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    array = cv2.resize(data, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC) / 255.
    return array


def get_raf_aligned(one_dir, one_txt):
    # 得到RAF aligned的npy文件
    dict_train, dict_test = {}, {}
    for line in open(one_txt, 'r'):
        line = line.strip()
        if line.startswith('train'):
            dict_train[line.split(' ')[0][:11] + '_aligned.jpg'] = int(line.split(' ')[1])
        elif line.startswith('test'):
            dict_test[line.split(' ')[0][:9] + '_aligned.jpg'] = int(line.split(' ')[1])
        else:
            print("wrong!!!")

    images, labels = [], []
    for k, v in dict_train.items():
        image_path = one_dir + k
        image = image2array(image_path)
        images.append(image)
        labels.append(rafnum2vec(v))
    for k, v in dict_test.items():
        image_path = one_dir + k
        images.append(image2array(image_path))
        labels.append(rafnum2vec(v))

    np.save(join(SAVE_DIRECTORY, 'raf_aligned_' + SAVE_DATASET_IMAGES_FILENAME), images)
    np.save(join(SAVE_DIRECTORY, 'raf_aligned_' + SAVE_DATASET_LABELS_FILENAME), labels)
    return "Get aligned!!!"


def get_raf_original(one_dir, one_txt):
    # 用opencv得到RAF original的npy文件
    images, labels, count = [], [], 0
    for line in open(one_txt, 'r'):
        line = line.strip()
        image_path = one_dir + line.split(' ')[0]
        image = format_image(cv2.imread(image_path), flag=1)
        if image is not None:
            count += 1
            images.append(image)
            labels.append(rafnum2vec(int(line.split(' ')[1])))

    print(count)  # 12717
    np.save(join(SAVE_DIRECTORY, 'sf=' + str(SCALEFACTOR) + '_raf_origin_' + SAVE_DATASET_IMAGES_FILENAME), images)
    np.save(join(SAVE_DIRECTORY, 'sf=' + str(SCALEFACTOR) + '_raf_origin_' + SAVE_DATASET_LABELS_FILENAME), labels)
    return "Get original!!!"


if __name__ == '__main__':
    get_fer(join(SAVE_DIRECTORY, DATASET_CSV_FILENAME))
    raf_aligned_dir = '/data1/emotion_rec/Real-world Affective Faces (RAF) Database/basic/Image/aligned/'
    raf_original_dir = '/data1/emotion_rec/Real-world Affective Faces (RAF) Database/basic/Image/original/'
    label_txt = '/data1/emotion_rec/Real-world Affective Faces (RAF) Database/basic/EmoLabel/list_patition_label.txt'
    get_raf_aligned(raf_aligned_dir, label_txt)
    get_raf_original(raf_original_dir, label_txt)
    pass
