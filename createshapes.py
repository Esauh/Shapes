import cv2 
import numpy as np 
import os
from os import listdir, makedirs
from os.path import isfile, join
import random
import glob


squarefilepath = '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/new-square'
hexafilepath = '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/hexagon'
pentagonfilepath = '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/pentagon'
heptagonfilepath = '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/heptagon'
irregularfilepath = '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/irregular'
openfilepath = '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/open'

def generate_hexagon(angle):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    vertices = np.array([[100, 20], [180, 70], [180, 150], [100, 200], [20, 150], [20, 70]])
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_vertices = cv2.transform(np.array([vertices]), M)[0]
    hexagon = cv2.polylines(img, [rotated_vertices], True, (0, 0, 0), thickness=2)
    return hexagon

def generate_square(angle, x, y):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    w, h = x * 2, y * 2
    x_offset = (img.shape[1] - w) // 2
    y_offset = (img.shape[0] - h) // 2
    vertices = np.array([[x_offset, y_offset], [x_offset + w, y_offset],
                         [x_offset + w, y_offset + h], [x_offset, y_offset + h]])
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

def generate_irregular_shape(angle, x, y):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    vertices = np.array([[x, y], [2*x, 2*y], [3*x, y], [3*x, 3*y], [x, 3*y]])
    noise = np.random.normal(0, 10, (5, 2)).astype(int)
    vertices += noise
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_vertices = cv2.transform(np.array([vertices]), M)[0]
    shape = cv2.polylines(img, [rotated_vertices], True, (0, 0, 0), thickness=2)
    return shape

def generate_open_shape(angle, x, y):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    vertices = np.array([[x, y], [2*x, 2*y], [3*x, y], [3*x, 3*y], [x, 3*y]])
    vertices = np.vstack((vertices, [2*x, y]))
    noise = np.random.normal(0, 10, (6, 2)).astype(int)
    vertices += noise
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_vertices = cv2.transform(np.array([vertices]), M)[0]
    shape = cv2.polylines(img, [rotated_vertices[:5]], False, (0, 0, 0), thickness=2)
    shape = cv2.line(shape, tuple(rotated_vertices[4]), tuple(rotated_vertices[5]), (0, 0, 0), thickness=2)
    return shape

# for i in range(3600):
#     angle = random.uniform(0.0,360.0)
#     x = random.randint(20, 75)
#     shape = generate_pentagon(angle)
#     filename = f"pentagon{i}.png"
#     cv2.imwrite(os.path.join(pentagonfilepath, filename), shape)


path_hexagon='/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/hexagon/*.png'
path_pentagon='/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/pentagon/*.png'
path_heptagon='/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/heptagon/*.png'
path_irregular='/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/irregular/*.png'
path_open= '/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/open/*.png'
paths_list = [path_open]
count = 14400
for path in paths_list:
    filenames = glob.glob(path)
    for filename in filenames:
        image = cv2.imread(filename)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filename = f"non-square{count}.png"
        cv2.imwrite(os.path.join("/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/non-square", filename),gray_img)
        count += 1

#If you mess up use this
# for filename in glob.glob("/Users/v.esau.hutcherson/codesrc/shapes/archive (4)/shapes/non-square/non-sqaure*"):
#     os.remove(filename) 

# # TODO: Fix this function vertices are not being accurately created using trig try and find out how to utilize trig for the creation of the shapes
# def generate_hexagon(angle, x):
#     img = np.ones((200, 200, 3), dtype=np.uint8) * 255
#     vertices_center = int (img.shape[0]//2)
#     vertices = np.array([[x, vertices_center], [-x, vertices_center], [int(x*math.cos(60)), int(x*math.sin(60))], [int(x*math.cos(120)), int(x*math.sin(120))], 
#                          [int(x*math.cos(240)), int(x*math.sin(240))], [int(x*math.cos(300)), int(x*math.sin(300))]])
#     center = (img.shape[1] // 2, img.shape[0] // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_vertices = cv2.transform(np.array([vertices]), M)[0]
#     hexagon = cv2.polylines(img, [rotated_vertices], True, (0, 0, 0), thickness=2)
#     return hexagon
