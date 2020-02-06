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

def CamFocalLengthXandWidthToAngleW(camFocalLengthX, camW):
    camAngleW = math.degrees(2*math.atan((camW/2)/camFocalLengthX))
    return camAngleW

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

## Camera Pixel and Ground Plane Intersection
def CamPixToPlaneIntersection(pixPos, camPos, camRot, camW, camH, camAngleW, planeNormal, planePoint):
    camPos = np.asarray(camPos)
    camRot = np.asarray(camRot)
    pixWorldPos = CamPixToWorldPos(pixPos, camPos, camRot, camW, camH, camAngleW)
    rayDirection = pixWorldPos - camPos
    rayPoint = camPos
    pt = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
    return pt


if __name__ == '__main__':

    ## Camera 01
    # camera setup
    camW = 1920  # units: pixel (mm when calculated)
    camH = 1080  # units: pixel (mm when calculated)
    camFocalLengthX = 1445.6323655009337 # units: pixel (mm when calculated); got from OpenCV camera calibratyion
    camAngleW = CamFocalLengthXandWidthToAngleW(camFocalLengthX, camW)  # units: degree
    print(f'camAngleW = {camAngleW} degrees')
    # camera pos rot
    camPos = [6239.727, -1172.106, 2909.551]  # units: mm  # optimized by RH GH Genetic Algorithm
    camRot = [-38.147, -2.203, 46.003]  # local euler angle(x, y, z)(p, r, y); units: degree  # optimized by RH GH Genetic Algorithm
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


    ## Camera 02
    # camera setup
    camW = 1920  # units: pixel (mm when calculated)
    camH = 1080  # units: pixel (mm when calculated)
    camFocalLengthX = 1445.6323655009337 # units: pixel (mm when calculated); got from OpenCV camera calibratyion
    camAngleW = CamFocalLengthXandWidthToAngleW(camFocalLengthX, camW)  # units: degree
    print(f'camAngleW = {camAngleW} degrees')
    # camera pos rot
    camPos = [-871.632, -1618.656, 2667.596]  # units: mm  # optimized by RH GH Genetic Algorithm
    camRot = [-39.545, 0.994, -53.561]  # local euler angle(x, y, z)(p, r, y); units: degree  # optimized by RH GH Genetic Algorithm
    # pixel info
    pixPos0 = [108.2696, 671.320829] # units: pixel (mm when calculated)
    pixPos1 = [733.570209, 288.436584] # units: pixel (mm when calculated)
    pixPos2 = [1174.086386, 457.519517] # units: pixel (mm when calculated)
    pixPos3 = [589.432408, 1028.874267] # units: pixel (mm when calculated)
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