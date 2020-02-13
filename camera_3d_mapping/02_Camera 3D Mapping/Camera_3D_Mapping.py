import numpy as np
import math

## Line-plane Intersction
def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
 
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

#Define plane
planeNormal = np.array([0, 0, 1])
planePoint = np.array([0, 0, 5]) #Any point on the plane

#Define ray
rayDirection = np.array([0, -1, -1])
rayPoint = np.array([0, 0, 10]) #Any point along the ray

Psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
print ("intersection at", Psi)

## Ray from Camera
# camera setup
camW = 1920  # units: pixel (mm when calculated)
camH = 1080  # units: pixel (mm when calculated)
camAngleW = 110  # units: degree

camD = (camW/2)/math.tan(math.radians(camAngleW/2))  # units: mm
print ('camD = ', camD)
camD =  672.1992366813215

# https://stackoverflow.com/questions/6802577/rotation-of-3d-vector

def RotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def RotPtAxis(pt, axis, theta):
    # theta in radians
    return np.dot(RotationMatrix(axis, theta), pt)

pt = [3, 5, 0]
axis = [4, 4, 1]
theta = 1.2 

print(RotPtAxis(pt, axis, theta))

print(math.degrees(1.2))

def CamFocalLengthXandWidthToAngleW(camFocalLengthX, camW):
    camAngleW = math.degrees(2*math.atan((camW/2)/camFocalLengthX))
    return camAngleW


camPos = [0, 0, 3000]  # units: mm
camRot = [-25, 5, 45]  # local euler angle(rhino coordinate system:x, y, z)(pitch, roll, yaw); units: degree

pixPos = [400, 600] # units: pixel (mm when calculated)

def CamPixToWorldPos(pixPos, camPos, camRot, camW, camH, camAngleW):
    '''
    camRot in degrees (operated in order of rhino coordinate system z,x,y/yaw, pitch, roll)
    pixPos in pixels (mm when calculated), [0,0]is top left of the image
    camPos in mm
    camW, camH in pixels (mm when calculated)
    camAngleW in degrees
    '''
    camD = (camW/2)/math.tan(math.radians(camAngleW/2))
    camPos = np.asarray(camPos)
    camRot = np.asarray(camRot)
    # start with cam at [0,0,0], fwd facing, no rot, get pix world pos
    pt = np.array([pixPos[0]-camW/2, camD, camH/2-pixPos[1]])
    # rotate pt with cam at [0,0,0]
    ## local z axis (yaw)
    localZAxis = [0,0,1]
    pt = RotPtAxis(pt, localZAxis, math.radians(camRot[2]))
    ## local x axis (pitch)
    localXAxis = [1,0,0]
    localXAxis = RotPtAxis(localXAxis, localZAxis, math.radians(camRot[2]))
    pt = RotPtAxis(pt, localXAxis, math.radians(camRot[0]))
    ## local y axis (roll)
    localYAxis = [0,1,0]
    localYAxis = RotPtAxis(localYAxis, localZAxis, math.radians(camRot[2]))
    localYAxis = RotPtAxis(localYAxis, localXAxis, math.radians(camRot[0]))
    pt = RotPtAxis(pt, localYAxis, math.radians(camRot[1]))
    # move pt to cam pos
    pt = pt + camPos
    return pt

print(CamPixToWorldPos(pixPos, camPos, camRot, camW, camH, camAngleW))

## Camera Pixel and Ground Plane Intersection
def CamPixToPlaneIntersection(pixPos, camPos, camRot, camW, camH, camAngleW, planeNormal, planePoint):
    camPos = np.asarray(camPos)
    camRot = np.asarray(camRot)
    pixWorldPos = CamPixToWorldPos(pixPos, camPos, camRot, camW, camH, camAngleW)
    rayDirection = pixWorldPos - camPos
    rayPoint = camPos
    pt = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
    return pt


## test 01
# camera setup
camW = 1920  # units: pixel (mm when calculated)
camH = 1080  # units: pixel (mm when calculated)
camAngleW = 110  # units: degree
# camera pos rot
camPos = [0, 0, 3000]  # units: mm
camRot = [-25, 5, 45]  # local euler angle(x, y, z); units: degree
# pixel info
pixPos = [400, 600] # units: pixel (mm when calculated)
# plane info
planeNormal = np.array([0, 0, 1])
planePoint = np.array([0, 0, 0]) #Any point on the plane
# test
pt = CamPixToPlaneIntersection(pixPos, camPos, camRot, camW, camH, camAngleW, planeNormal, planePoint)
print(f'intersection point: {pt[0]}, {pt[1]}, {pt[2]}')


## test 02
# camera setup
camW = 800  # units: pixel (mm when calculated)
camH = 600  # units: pixel (mm when calculated)
camAngleW = 95  # units: degree
# camera pos rot
camPos = [5503, 2405, 2837]  # units: mm
camRot = [-21.9, -12.5, -45]  # local euler angle(x, y, z); units: degree
# pixel info
pixPos = [135, 402] # units: pixel (mm when calculated)
# plane info
planeNormal = np.array([0, 0, 1])
planePoint = np.array([0, 0, 0]) #Any point on the plane
# test
pt = CamPixToPlaneIntersection(pixPos, camPos, camRot, camW, camH, camAngleW, planeNormal, planePoint)
print(f'intersection point: {pt[0]}, {pt[1]}, {pt[2]}')


## test 03
# camera setup
camW = 1920  # units: pixel (mm when calculated)
camH = 1080  # units: pixel (mm when calculated)
camFocalLengthX = 1445.6323655009337 # units: pixel (mm when calculated); got from OpenCV camera calibratyion
camAngleW = CamFocalLengthXandWidthToAngleW(camFocalLengthX, camW)  # units: degree
print(f'camAngleW = {camAngleW} degrees')
# camera pos rot
camPos = [6230.8993599197765, -1183.7222483342364, 2889.166547540668]  # units: mm
camRot = [-37.83221208036811, -1.5177005471375555, 47.150730747104895]  # local euler angle(x, y, z)(p, r, y); units: degree
# pixel info
pixPos0 = [835.556383, 357.367298] # units: pixel (mm when calculated)
pixPos1 = [1566.988522, 812.879272] # units: pixel (mm when calculated)
pixPos2 = [1252.272453, 1064.874888] # units: pixel (mm when calculated)
pixPos3 = [271.078929, 226.752521] # units: pixel (mm when calculated)
# plane info
planeNormal = np.array([0, 0, 1])
planePoint = np.array([0, 0, 0]) #Any point on the plane
# test
pt0 = CamPixToPlaneIntersection(pixPos0, camPos, camRot, camW, camH, camAngleW, planeNormal, planePoint)
print(f'intersection point 0: {pt0[0]}, {pt0[1]}, {pt0[2]}')
pt1 = CamPixToPlaneIntersection(pixPos1, camPos, camRot, camW, camH, camAngleW, planeNormal, planePoint)
print(f'intersection point 1: {pt1[0]}, {pt1[1]}, {pt1[2]}')
pt2 = CamPixToPlaneIntersection(pixPos2, camPos, camRot, camW, camH, camAngleW, planeNormal, planePoint)
print(f'intersection point 2: {pt2[0]}, {pt2[1]}, {pt2[2]}')
pt3 = CamPixToPlaneIntersection(pixPos3, camPos, camRot, camW, camH, camAngleW, planeNormal, planePoint)
print(f'intersection point 3: {pt3[0]}, {pt3[1]}, {pt3[2]}')


