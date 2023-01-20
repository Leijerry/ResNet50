import os
import cv2
from MTCNN import MTCNN

outer_path = '/srv/data1/arunirc/datasets/vggface2'
filelist = os.listdir(outer_path)

detector = MTCNN()

for item in filelist:
    src = os.path.join(os.path.abspath(outer_path), item)
    input_img = cv2.imread(src)

    detected = detector.detect_faces(input_img)
    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, w, h = d['box']
            x2 = x1 + w
            y2 = y1 + h
            image = input_img[(y1 - 10):(y2 + 10), (x1 - 10):(x2 + 10)]
            cv2.imwrite(src, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

