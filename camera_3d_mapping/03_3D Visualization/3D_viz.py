# imports
from vpython import *

import sys
sys.path.extend(['../02_Camera 3D Mapping/', '../01_Camera Pose Estimation/'])
from convert_stl import *
from Camera_3D_Mapping_clear import *


# functions
def Lst2Vec(list):
    '''
    list requires 3 elements
    '''
    return vector(list[0], list[1], list[2])

def lineWithBox(pt1, pt2, thickness, color):
    ptv = pt2-pt1
    ptm = (pt1+pt2)/2
    myBox = box(pos=ptm, axis=pt2-pt1, length=ptv.mag, height=thickness, width=thickness, color=color)
    return myBox


# main
if __name__ == '__main__':

    # vpython scene
    scene = canvas(title='', width=1600, height=900, center=vector(0,0,0), background=color.black)


    # # plot vpython axises
    # curve(pos=[vector(0,0,0), vector(1500,0,0)], color=color.red, radius=7.5)  # x
    # curve(pos=[vector(0,0,0), vector(0,1500,0)], color=color.green, radius=7.5)  # y
    # curve(pos=[vector(0,0,0), vector(0,0,1500)], color=color.blue, radius=7.5)  # z


    # import world 3d model
    w = stl_to_triangles('t02.stl')
    w.color = color.white
    w.rotate(angle=-0.5*pi, axis=vector(1,0,0))

    # example: world pos <> compound pos
    world_pos = w.compound_to_world(vector(0,0,1))
    #print(f'world_pos: {world_pos}')
    compound_pos = w.world_to_compound(vector(0,0,1))
    #print(f'compound_pos: {compound_pos}')

    '''
    from here on, to viz coordinate in Rhino coordinate system (z up), 
    w.compound_to_world(pt_z_up) is a necessary conversion
    '''

    # plot compound/rhino axises
    curve(pos=[w.compound_to_world(vector(0,0,0)), w.compound_to_world(vector(1000,0,0))], color=color.red, radius=10)  # x
    curve(pos=[w.compound_to_world(vector(0,0,0)), w.compound_to_world(vector(0,1000,0))], color=color.green, radius=10)  # y
    curve(pos=[w.compound_to_world(vector(0,0,0)), w.compound_to_world(vector(0,0,1000))], color=color.blue, radius=10)  # z

    # camera setup
    ## Camera 01
    camW = 1920  # units: pixel (mm when calculated)
    camH = 1080  # units: pixel (mm when calculated)
    camFocalLengthX = 1445.6323655009337 # units: pixel (mm when calculated); got from OpenCV camera calibratyion
    camAngleW = CamFocalLengthXandWidthToAngleW(camFocalLengthX, camW)  # units: degree
    # camera pos rot
    camPos = [6239.727, -1172.106, 2909.551]  # units: mm  # optimized by RH GH Genetic Algorithm
    camRot = [-38.147, -2.203, 46.003]  # local euler angle(x, y, z)(p, r, y); units: degree  # optimized by RH GH Genetic Algorithm

    # viz camera
    # draw a camera at origin, forward facing
    camVizObjs = []
    camVizObjs.append(lineWithBox(w.compound_to_world(vector(0,0,0)), w.compound_to_world(vector(0,camFocalLengthX,0)), 5, color.green))
    camVizObjs.append(lineWithBox(w.compound_to_world(vector(0,0,0)), w.compound_to_world(vector(-camW/2,camFocalLengthX,-camH/2)), 5, color.green))
    camVizObjs.append(lineWithBox(w.compound_to_world(vector(0,0,0)), w.compound_to_world(vector(camW/2,camFocalLengthX,-camH/2)), 5, color.green))
    camVizObjs.append(lineWithBox(w.compound_to_world(vector(0,0,0)), w.compound_to_world(vector(camW/2,camFocalLengthX,camH/2)), 5, color.green))
    camVizObjs.append(lineWithBox(w.compound_to_world(vector(0,0,0)), w.compound_to_world(vector(-camW/2,camFocalLengthX,camH/2)), 5, color.green))
    camVizObjs.append(lineWithBox(w.compound_to_world(vector(-camW/2,camFocalLengthX,-camH/2)), w.compound_to_world(vector(camW/2,camFocalLengthX,-camH/2)), 5, color.green))
    camVizObjs.append(lineWithBox(w.compound_to_world(vector(camW/2,camFocalLengthX,-camH/2)), w.compound_to_world(vector(camW/2,camFocalLengthX,camH/2)), 5, color.green))
    camVizObjs.append(lineWithBox(w.compound_to_world(vector(camW/2,camFocalLengthX,camH/2)), w.compound_to_world(vector(-camW/2,camFocalLengthX,camH/2)), 5, color.green))
    camVizObjs.append(lineWithBox(w.compound_to_world(vector(-camW/2,camFocalLengthX,camH/2)), w.compound_to_world(vector(-camW/2,camFocalLengthX,-camH/2)), 5, color.green))
    camVizCompound = compound(camVizObjs, origin=w.compound_to_world(vector(0,0,0)), pos=w.compound_to_world(vector(0,0,0)), axis=w.compound_to_world(vector(1,0,0)))
    # rotate the camera in order of rhino coordinate system z,x,y/yaw, pitch, roll
    ## local z axis (yaw)
    localZAxis = [0,0,1]
    camVizCompound.rotate(angle=math.radians(camRot[2]), axis=w.compound_to_world(Lst2Vec(localZAxis)))
    ## local x axis (pitch)
    localXAxis = [1,0,0]
    localXAxis = RotPtAxis(localXAxis, [0,0,1], math.radians(camRot[2]))
    camVizCompound.rotate(angle=math.radians(camRot[0]), axis=w.compound_to_world(Lst2Vec(localXAxis)))
    ## local y axis (roll)
    localYAxis = [0,1,0]
    localYAxis = RotPtAxis(localYAxis, localZAxis, math.radians(camRot[2]))
    localYAxis = RotPtAxis(localYAxis, localXAxis, math.radians(camRot[0]))
    camVizCompound.rotate(angle=math.radians(camRot[1]), axis=w.compound_to_world(Lst2Vec(localYAxis)))
    # move the camera to camPos
    camVizCompound.pos = w.compound_to_world(Lst2Vec(camPos))


    # viz foot pts in 3d
    # pixel info
    footPixPts =    [
                        [1112.712663, 639.248277], 
                        [1158.556281, 667.315499], 
                        [1173.863035, 705.305794], 
                        [1075.067027, 660.718376]
                    ] # units: pixel (mm when calculated)

    averageFootPixPt = [1135.634472, 653.281888]
    # plane info
    groundPlanePoint = np.array([0, 0, 0]) #Any point on the plane
    groundPlaneNormal = np.array([0, 0, 1])
    # calc
    footWorldPts = []
    for footPixPt in footPixPts:
        footWorldPts.append(CamPixToPlaneIntersection(footPixPt, camPos, camRot, camW, camH, camAngleW, groundPlaneNormal, groundPlanePoint))
    #print(f'footWorldPts: {footWorldPts}')
    averageFootWorldPt = CamPixToPlaneIntersection(averageFootPixPt, camPos, camRot, camW, camH, camAngleW, groundPlaneNormal, groundPlanePoint)
    #print(f'averageFootWorldPt: {averageFootWorldPt}')
    # viz
    vizObjs = []
    # foot pts
    for footWorldPt in footWorldPts:
        vizObjs.append(sphere(pos=w.compound_to_world(Lst2Vec(footWorldPt)), color=color.magenta, radius=30))
    # average foot pt
    vizObjs.append(sphere(pos=w.compound_to_world(Lst2Vec(averageFootWorldPt)), color=color.red, radius=30))

    # viz body pts in 3d
    # pixel info
    bodyPixPts =    [
                        [1139.854535, 237.401374], 
                        [1114.789456, 183.671035], 
                        [1150.051027, 183.397576], 
                        [1067.619219, 235.728285], 
                        [1226.390776, 245.358857], 
                        [1041.912619, 353.602978], 
                        [1233.358995, 380.900494], 
                        [1083.190521, 427.801254], 
                        [1170.182089, 421.660546], 
                        [1037.472876, 444.386003], 
                        [1195.343392, 450.213863], 
                        [1094.897208, 539.073642], 
                        [1167.930603, 567.195150]
                    ] # units: pixel (mm when calculated)# plane info

    # plane info
    bodyPlanePoint = averageFootWorldPt
    camProjToGround = [camPos[0], camPos[1], 0]
    bodyPlaneNormal = camProjToGround - bodyPlanePoint
    #print(f'bodyPlaneNormal: {bodyPlaneNormal}')
    # calc
    bodyWorldPts = []
    for bodyPixPt in bodyPixPts:
        bodyWorldPts.append(CamPixToPlaneIntersection(bodyPixPt, camPos, camRot, camW, camH, camAngleW, bodyPlaneNormal, bodyPlanePoint))
    #print(f'bodyWorldPts: {bodyWorldPts}')
    # viz
    # body pts
    for bodyWorldPt in bodyWorldPts:
        vizObjs.append(sphere(pos=w.compound_to_world(Lst2Vec(bodyWorldPt)), color=color.orange, radius=30))