# -*- coding:utf-8 -*-

import cv2
from constants import *
from train import EmotionRecognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

global face
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
eye_cascade = cv2.CascadeClassifier(EYEGLASSSES_CASC)

# Load Model
network = EmotionRecognition()
network.build_network()


def format_image(image):
    global face

    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for oneface in faces:
        if oneface[2] * oneface[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = oneface

    # draw face
    face = max_area_face
    x, y, w, h = face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # draw eyes
    roi_gray = image[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    # Chop image to face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    # Resize image to network size
    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None

    return image


if __name__ == '__main__':
    mode = 'cam'  # 'cam' default,or you can add a video path
    if mode == 'cam':  # camera
        video_capture = cv2.VideoCapture(0)
    else:  # video
        video_capture = cv2.VideoCapture(mode)
    
    # 设定摄像头分辨率
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)

    feelings_faces = []

    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('face.mp4', fourcc, 10, (640, 480))
    fps_time = 0

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Predict result with network
        result = network.predict(format_image(frame))

        # Write results in frame
        if result is not None:
            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(frame, emotion, (10, index * 20 + 20),
                            cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100),
                                                              (index + 1) * 20 + 4), (255, 0, 0), -1)

            face_image = feelings_faces[np.argmax(result[0])]

            # Ugly transparent fix
            for c in range(0, 3):
                frame[200:320, 10:130, c] = face_image[:, :, c] * \
                                            (face_image[:, :, 3] / 255.0) + frame[200:320,
                                                                            10:130, c] * (
                                                                            1.0 - face_image[:, :, 3] / 255.0)
            # draw emotion text
            text = EMOTIONS[np.argmax(result[0])]
            # print(result, text)
            cv2.putText(frame, text, (face[0], face[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

            # put CH name on frames
            img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # 字体 字体*.ttc的存放路径一般是： /usr/share/fonts/opentype/noto/ 查找指令locate *.ttc
            img_font = ImageFont.truetype('NotoSansCJK-Bold.ttc', 20)
            fillColor = (0, 255, 0)
            position = (face[0]+face[2]-40, face[1]-25)
            draw = ImageDraw.Draw(img_PIL)
            ch_str = '中国'
            draw.text(position, ch_str, font=img_font, fill=fillColor)
            frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        
        # FPS   
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    
        out.write(frame)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        fps_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
