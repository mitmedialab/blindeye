import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt

img_input = cv2.imread('example.png')
img_output = cv2.imread('example-disparity-map.png')

new_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
new_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)
disparity = stereo.compute(new_input, new_output)

print(disparity)

cv2.imshow("disparity", disparity)
cv2.waitKey(0)
