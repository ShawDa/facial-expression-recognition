# -*- coding:utf-8 -*-
#                               __                    __
#                              /\ \__                /\ \__
#   ___    ___     ___     ____\ \ ,_\    __      ___\ \ ,_\   ____
#  /'___\ / __`\ /' _ `\  /',__\\ \ \/  /'__`\  /' _ `\ \ \/  /',__\
# /\ \__//\ \L\ \/\ \/\ \/\__, `\\ \ \_/\ \L\.\_/\ \/\ \ \ \_/\__, `\
# \ \____\ \____/\ \_\ \_\/\____/ \ \__\ \__/.\_\ \_\ \_\ \__\/\____/
#  \/____/\/___/  \/_/\/_/\/___/   \/__/\/__/\/_/\/_/\/_/\/__/\/___/  .txt
#
#

CASC_PATH = 'haarcascades/haarcascade_frontalface_default.xml'
EYE_CASC = 'haarcascades/haarcascade_eye.xml'
EYEGLASSSES_CASC = 'haarcascades/haarcascade_eye_tree_eyeglasses.xml'
SIZE_FACE = 48
SCALEFACTOR = 1.1
EMOTIONS = ['angry', 'disgusted', 'fearful','happy', 'sad', 'surprised', 'neutral']
RAF_EMOTIONS = ['surprised', 'fearful', 'disgusted', 'happy', 'sad', 'angry', 'neutral']
SAVE_DIRECTORY = 'data'
SAVE_MODEL_FILENAME = 'Gudi_model_100_epochs_20000_faces'
DATASET_CSV_FILENAME = 'fer2013.csv'
SAVE_DATASET_IMAGES_FILENAME = 'data_images.npy'
SAVE_DATASET_LABELS_FILENAME = 'data_labels.npy'