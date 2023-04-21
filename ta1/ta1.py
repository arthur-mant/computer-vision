import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

#im_path = './img.jpg'


direc = "./images_output/"
os.makedirs(direc, exist_ok = True)

im_path = './img.jpeg'

down_sampling = 2

img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
for i in range(down_sampling):
    img = cv2.pyrDown(img)

cv2.imwrite(direc+"downsampled.jpeg", img)

#kernel = np.ones([kernel_size, kernel_size], np.float32)/(kernel_size*kernel_size)
kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
smooth = cv2.filter2D(img,-1,kernel)

cv2.imwrite(direc+"smooth.jpeg", smooth)

detail = img-smooth

cv2.imwrite(direc+"detail.jpeg", detail)

kernel = np.ones([3, 3], np.float32)/9
detail_blurred = cv2.filter2D(detail,-1,kernel)

cv2.imwrite(direc+"detail_blurred.jpeg", detail_blurred)

ret, detail_clean = cv2.threshold(detail_blurred, 127, 255, cv2.THRESH_TOZERO)

cv2.imwrite(direc+"detail_clean.jpeg", detail_clean)

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(detail_clean, -1, kernel)

cv2.imwrite(direc+"sharpened_final.jpeg", sharpened)

canny = cv2.Canny(img, 20, 40)

cv2.imwrite(direc+"canny.jpeg", canny)

