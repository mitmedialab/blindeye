import numpy as np
import math

camToObjRotEuler = np.array([-2.2497639125491067, 0.021106987097628618, -0.9365709689969954])

# functions to get angle between 2 vectors
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def signed_angle(v1, v2, vn):
    '''
    works in 3D
    v1: from vector
    v2: to vector
    vn: norm of the plane v1 and v2 lie in (right-hand)
    '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    cross = np.cross(v1, v2);
    angle *= np.sign(np.dot(vn,cross))
    return angle

# functions to rotate point/vector around axis
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

# main function
def rotXYZtoPRY(camRotXYZ):
    '''
    camRotXYZ in radius (in the order of rhino coordinate system x, y, z axises)
    camRotPRY in degrees (order: pitch, roll, yaw (rhino x y z). operated in order of rhino coordinate system z,x,y/yaw, pitch, roll)
    '''
    # Camera uses opencv coordinate system, z is forward. 
    # To start with, in Rhino coordinate system, camera z/forward is up. 
    # First we need to rotate the cam z 3 times according to camRotXYZ to get rotated cam z
    worldX = [1,0,0]
    worldY = [0,1,0]
    worldZ = [0,0,1]
    camZ = [0,0,1]
    camZ = RotPtAxis(camZ, worldX, camRotXYZ[0])
    camZ = RotPtAxis(camZ, worldY, camRotXYZ[1])
    camZ = RotPtAxis(camZ, worldZ, camRotXYZ[2])
    print(f'camZ: {camZ[0]}, {camZ[1]}, {camZ[2]}')
    camX = [1,0,0] # will be used to decide sign of angle
    camX = RotPtAxis(camX, worldX, camRotXYZ[0])
    camX = RotPtAxis(camX, worldY, camRotXYZ[1])
    camX = RotPtAxis(camX, worldZ, camRotXYZ[2])
    print(f'camX: {camX[0]}, {camX[1]}, {camX[2]}')

    # then get pitch of rotated camera by get the angle between rotated z and z projected to xy plane
    camZprojXY = [camZ[0],camZ[1],0]
    #print(f'camZprojXY: {camZprojXY}')
    pitch = math.degrees(signed_angle(camZprojXY, camZ, camX))
    print(f'pitch: {pitch} degrees')
    # get yaw of camera by get the angle between world y and z projected
    yaw = math.degrees(signed_angle(worldY, camZprojXY, worldZ))
    print(f'yaw: {yaw} degrees')

    # to get roll of camera, first get rotated negative y axis (camera up) of camera opencv coordinate system
    camNegY = [0,-1,0]
    camNegY = RotPtAxis(camNegY, worldX, camRotXYZ[0])
    camNegY = RotPtAxis(camNegY, worldY, camRotXYZ[1])
    camNegY = RotPtAxis(camNegY, worldZ, camRotXYZ[2])
    print(f'camNegY: {camNegY[0]}, {camNegY[1]}, {camNegY[2]}')
    camZ
    # then rotate the camNegY yaw degree along world z
    camNegY = RotPtAxis(camNegY, worldZ, math.radians(-yaw))
    print(f'camNegY: {camNegY[0]}, {camNegY[1]}, {camNegY[2]}')
    # rotate the camNegY pitch degree along world x, to make camNegY in world xy plane
    camNegY = RotPtAxis(camNegY, worldX, math.radians(-pitch))
    print(f'camNegY: {camNegY[0]}, {camNegY[1]}, {camNegY[2]}')
    # get roll of camera by get the angle between world z and camNegY
    roll = math.degrees(signed_angle(worldZ, camNegY, worldY))
    print(f'roll: {roll} degrees')

    # combine pitch, roll, yaw to camRotYPR
    camRotPRY = [pitch, roll, yaw]
    return camRotPRY

camRotPRY = rotXYZtoPRY(camToObjRotEuler)
print(f'camRotPRY: {camRotPRY[0]}, {camRotPRY[1]}, {camRotPRY[2]} degrees')