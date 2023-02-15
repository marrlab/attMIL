import olefile
from PIL import Image
import cv2
import numpy as np


def zviloader(path):
    stream_label = ['Image', 'Item(1)', 'Contents']
    ole = olefile.OleFileIO(path)

    data = ole.openstream(stream_label).read()
    img = Image.frombytes('I;16L', (1300, 1030), data)
    arr = np.roll(np.asarray(img, dtype=np.uint16), -162)
    # img = np.asarray(img,dtype=np.uint16)
    norm_image = cv2.normalize(arr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = cv2.normalize(norm_image,None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    norm_image = cv2.resize(norm_image, (572, 572))
    norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2BGR)

    return norm_image
