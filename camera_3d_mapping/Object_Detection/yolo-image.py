import numpy as np
import argparse
import time
import cv2
from PIL import Image
import os
## Change os to Path

def getDepths(image, model):
	from keras.models import load_model
	from layers import BilinearUpSampling2D
	from utils import predict, load_images, display_images

	custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

	print('Loading model...')

	# Load model into GPU / CPU
	model = load_model(model, custom_objects=custom_objects, compile=False)
	print('\nModel loaded ({0}).'.format(model))

	# Input images
	inputs = load_images(image)
	print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))
	#print(inputs.shape) = original size

	# Compute results
	outputs = predict(model, inputs)

	np.save('3D-mappings',outputs)

	return outputs

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
ap.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
args = vars(ap.parse_args())

labelspath = os.path.sep.join([args["yolo"],"coco.names"])
LABELS = open(labelspath).read().strip().split("\n")

print('here',str(args))

np.random.seed(10)
COLORS = np.random.randint(0,255,size=(len(LABELS),3),dtype='uint8')

weightsPath = os.path.sep.join([args["yolo"],"yolov3.weights"])
configPath = os.path.sep.join([args["yolo"],"yolov3.cfg"])

print("Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load input image and grab its spatial dimensions
image = cv2.imread(args["image"])
if image.shape[:2] != (240,320):
	image = cv2.resize(image,(240,320))
(H, W) = image.shape[:2]
print((H,W))

# get 3D data
data = getDepths(args["image"], args['model'])
depths = data[0]
## ***OUTDATED WAY***
# data = np.load('../3DReconstruction/DenseDepth/3D-mappings.npy')
# depths = data[0]
# #depths = data[...,2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))


# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > args['confidence']:
            box = detection[0:4] * np.array([W,H,W,H])
            (centerX, centerY, width, height) = box.astype("int")

            x = int(centerX - (width/2))
            y = int(centerY - (height/2))

            boxes.append([x,y,int(width),int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# write bounding box, depth information to a file
f = open('objects-found.txt','w+')
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# find average depth of object from center
		centerX = int(x + w/2)
		centerY = int(y + h/2)
		depth_sum = 0
		center_pixels = 0

		if (centerX-int(w/16))-(centerX+int(w/16)) == 0 and (centerY-int(h/16))-(centerY+int(h/16)) == 0:
				depth_sum = depths[centerX][centerY][0]
		elif (centerX-int(w/16))-(centerX+int(w/16)) == 0:
			for y_h in range(centerY-int(h/16),centerY+int(h/16)):
				depth_sum += depths[centerX][y_h][0]
				center_pixels += 1
		elif (centerY-int(h/16))-(centerY+int(h/16)) == 0:
			for x_w in range(centerX-int(w/16),centerX+int(w/16)):
				depth_sum += depths[x_w][centerY][0]
				center_pixels += 1
		else:
			for x_w in range(centerX-int(w/16),centerX+int(w/16)):
				for y_h in range(centerY-int(h/16),centerY+int(h/16)):
					depth_sum += depths[x_w][y_h][0]
					center_pixels += 1

		if center_pixels == 0:
			center_pixels += 1

		depth = (depth_sum * 1.0) / center_pixels

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f} \n z: {:.4f}".format(LABELS[classIDs[i]], confidences[i],depth)
		f.write(text + '\n')
		print(text)
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.35, color, 2)

# show and write the output image
cv2.imshow("Image", image)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('Outputs/'+args['image'].replace('.png','-output.png').replace('Inputs/',''), image)
    cv2.destroyAllWindows()
