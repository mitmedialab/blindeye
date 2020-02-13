import numpy as np
import cv2 as cv
import glob


# Load previously saved camera calibration data
with np.load(r'..\00_Camera Calibration\data\calib.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


def draw1(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw2(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in cayon
    img = cv.drawContours(img, [imgpts[:4]],-1,(255,255,0),-3)
    # # draw pillars in blue color
    # for i,j in zip(range(4),range(4,8)):
    #     img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # # draw top layer in red color
    # img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


a = 8
b = 7
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((b*a,3), np.float32)
objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
print(f'objp: {objp}')

# Render a Cube
for fname in glob.glob(r'..\00_Camera Calibration\input images\*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (a,b), None)
    print(f'corners: {corners}')
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        print(f'corners2: {corners2}')
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        print(f'tvecs: {tvecs}; \nrvecs: {rvecs}')
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        #img = draw2(img,corners2,imgpts)
        img = draw1(img,corners2,imgpts)
        cv.imshow('img',img)
        # if 's' key pressed, save result image
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(f'{fname[:-4]}_pose_estimation.jpg', img)
cv.destroyAllWindows()





axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])