import numpy as np
import cv2 as cv
import glob
import math


# Load previously saved camera calibration data
with np.load(r'..\00_Camera Calibration\data\calib.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

print(f'mtx: {mtx}')
print(f'dist: {dist}')

camFocalLengthX = mtx[0,0]
print(f'camFocalLengthX: {camFocalLengthX} pixel')


objPts_cam = np.array(  [
                            [0.0, 1828.8, 0.0], 
                            [2438.4, 1828.8, 0.0], 
                            [2438.4, 0.0, 0.0], 
                            [0.0, 0.0, 0.0]
                        ] )


corners_cam = np.array( [
                            [108.2696, 671.320829], 
                            [733.570209, 288.436584], 
                            [1174.086386, 457.519517], 
                            [589.432408, 1028.874267]
                        ] )


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


# Calculate camera translation and rotation
ret, rvecs, tvecs = cv.solvePnP(objPts_cam, corners_cam, mtx, dist)
print(f'----------------\ntvecs: {tvecs}; \nrvecs: {rvecs}')
objToCamRotM = cv.Rodrigues(rvecs)[0]
camToObjPosition = -np.matrix(objToCamRotM).T * np.matrix(tvecs)
print(f'----------------\ncamToObjPosition: {camToObjPosition[0,0]}, {camToObjPosition[1,0]}, {camToObjPosition[2,0]}')
#objToCamRotEuler = rotationMatrixToEulerAngles(objToCamRotM) # working
#print(f'objToCamRotEuler: {objToCamRotEuler}\n(Rotate objects in the order of rhino coordinate system x, y, z axises)') 
camToObjRotM = cv.Rodrigues(-rvecs)[0]
camToObjRotEuler = rotationMatrixToEulerAngles(camToObjRotM) # working
print(f'camToObjRotEuler: {camToObjRotEuler[0]}, {camToObjRotEuler[1]}, {camToObjRotEuler[2]}\n(Rotate camera in the order of rhino coordinate system x, y, z axises)')