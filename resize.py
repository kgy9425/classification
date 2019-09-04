import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

train_path = r'D:\CWA4호기\resize\SURFI4\NG_Body\train'

path = os.path.join(train_path, '*.bmp')
files = glob.glob(path)
i = 1

for fl in files:
    print(fl)
    image = cv2.imread(fl)
    try:
        image = cv2.resize(image, (224, 224), cv2.INTER_LINEAR)
        cv2.imwrite(fl, image)
        i = i+1
    except Exception as e:
        print(str(e))