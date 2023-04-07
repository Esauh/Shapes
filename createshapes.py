import cv2 
import numpy as np 
import os
from os import listdir
from os.path import isfile, join


squarefilepath = '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/new-square'
hexafilepath = '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/hexagon'
pentagonfilepath = '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/pentagon'
heptagonfilepath = '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/heptagon'

black = (0,0,0)



def generate_hexagon(angle):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    vertices = np.array([[100, 20], [180, 70], [180, 150], [100, 200], [20, 150], [20, 70]])
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_vertices = cv2.transform(np.array([vertices]), M)[0]
    hexagon = cv2.polylines(img, [rotated_vertices], True, (0, 0, 0), thickness=2)
    return hexagon

def generate_square(angle):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    x, y, w, h = 50, 50, 100, 100
    vertices = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_vertices = cv2.transform(np.array([vertices]), M)[0]
    square = cv2.polylines(img, [rotated_vertices], True, (0, 0, 0), thickness=2)

    return square

def generate_pentagon(angle):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    x, y, w, h = 50, 50, 100, 100
    pent_vertices = np.array([[x+w//2, y], [x+w, y+h//3], [x+w*3//4, y+h], [x+w//4, y+h], [x, y+h//3]])
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_pent_vertices = cv2.transform(np.array([pent_vertices]), M)[0]
    pentagon = cv2.polylines(img, [rotated_pent_vertices], True, (0, 0, 0), thickness=2)
    return pentagon



def generate_heptagon(angle):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    x, y, w, h = 30, 30, 140, 140
    hept_vertices = np.array([[x+w//2, y], [x+w*7//8, y+h//4], [x+w*5//6, y+h*3//4], [x+w//2, y+h], [x+w//6, y+h*3//4], [x+w//4, y+h//2], [x+w//8, y+h//4]])
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_hept_vertices = cv2.transform(np.array([hept_vertices]), M)[0]
    heptagon = cv2.polylines(img, [rotated_hept_vertices], True, (0, 0, 0), thickness=2)
    return heptagon

if not os.path.exists(heptagonfilepath):
    os.makedirs(heptagonfilepath)

for i in range(72):
    angle = i * 5
    shape = generate_heptagon(angle)
    filename = f"heptagon{i}.png"
    cv2.imwrite(os.path.join(heptagonfilepath, filename), shape)
