import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
a = 8
b = 7
objp = np.zeros((b*a,3), np.float32)
objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(r'input images\*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    cv.imshow('img', gray)
    cv.waitKey(10)
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (a,b), None)
    
    cv.imshow('img', img)
    cv.waitKey(10)
    
    # If found, add object points, image points (after refining them)
    print(f'image: {fname}; ret: {ret}')
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (a,b), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez(r'data\calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


# # Undistortion
# img = cv.imread(r'C:\OpenCV\OpenCV_Python\T001_20190703021430_Snapshot.jpg')
# h,  w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# # Using cv.undistort()
# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# # crop the image
# #x, y, w, h = roi
# #dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibresult.png', dst)

# '''
# # Using remapping
# # undistort
# mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite(r'C:\OpenCV\OpenCV_Python\calibresult.png', dst)
# '''