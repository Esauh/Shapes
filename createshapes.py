import cv2 
import numpy as np 
import os
from os import listdir
from os.path import isfile, join


filepath = '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/created-non-square'


def generate_shape(angle):
    shape = np.ones((200, 200, 3), dtype=np.uint8)
    shape = 255 *shape
    M = cv2.getRotationMatrix2D((10, 10), angle, 1.0)
    shape = cv2.warpAffine(shape, M, (200, 200), borderValue=0)
    return shape

if not os.path.exists(filepath):
    os.makedirs(filepath)

for i in range(10):
    angle = i * 5
    shape = generate_shape(angle)
    filename = f"rectangle{i}.png"
    cv2.imwrite(os.path.join(filepath, filename), shape)
