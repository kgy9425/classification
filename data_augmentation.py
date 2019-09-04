
import cv2
import os
import glob

train_path = r'D:\classification\data\train\imul2'
path = os.path.join(train_path, '*.BMP')
files = glob.glob(path)
i = 61
for fl in files:
    image = cv2.imread(fl)
    image2 = cv2.flip(image, 0)
    cv2.imwrite(repr(i) + '.BMP', image2)
    i = i + 1

