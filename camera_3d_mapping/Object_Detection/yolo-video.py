import numpy as np
import argparse
import imutils
import time
import cv2
import os # change to Path
from yolo_image import doesOverlap, getDepths,objectDepth, loadModel

# terminal arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
# 	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
ap.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
args = vars(ap.parse_args())

# load the COCO class labels
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# colors for each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load YOLO object detector
# determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# load model
model = loadModel(args['model'])

# initialize the video stream
vs = cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)


# try to determine the total number of frames in the video file
# try:
# 	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
# 		else cv2.CAP_PROP_FRAME_COUNT
# 	total = int(vs.get(prop))
# 	print("[INFO] {} total frames in video".format(total))
#
# except:
# 	print("[INFO] could not determine # of frames in video")
# 	print("[INFO] no approx. completion time can be provided")
# 	total = -1

# loop over frames from the video file stream
frames = 0
while True:
	frames += 1
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, end of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if frame.shape[:2] != (240,320):
		frame = cv2.resize(frame,(240,320))
		(H, W) = frame.shape[:2]

	# get 3D data
	data = getDepths(frame,model)
	depths = data[0]

	#print((H,W))

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

			# apply non-maxima suppression to suppress weak, overlapping bounding boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
				args["threshold"])

			# write bounding box, depth information to a file
			# f = open('objects-found.txt','a+')
			# f.write(args['input'] + '\n' + '\n')
			avg_depths = []
			bounding_boxes = []
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					depth = objectDepth(x,y,w,h,depths)
					avg_depths.append(depth)
					bounding_boxes.append(boxes[i])

					# draw a bounding box rectangle and label on the image
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f} \n z: {:.4f}".format(LABELS[classIDs[i]], confidences[i],depth)
					# f.write(text + '\n' + 'x: %d, y: %d' % (x,y) + '\n')
					# print(text)
					cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
						0.35, color, 2)

			# f.write('--------------------------------' + '\n')

			for i in range(len(bounding_boxes)):
				for j in range(i+1,len(bounding_boxes)):
					(x1, y1, w1, h1) = (bounding_boxes[i][0],bounding_boxes[i][1],bounding_boxes[i][2],bounding_boxes[i][3])
					(x2, y2, w2, h2) = (bounding_boxes[j][0],bounding_boxes[j][1],bounding_boxes[j][2],bounding_boxes[j][3])

					l1 = Point(x1, y1)
					r1 = Point(x1 + w1, y1 + h1)
					l2 = Point(x2, y2)
					r2 = Point(x2 + w2, y2 + h2)

					if doesOverlap(l1, r1, l2, r2):
						if abs(avg_depths[i]-avg_depths[j]) <= 0.03:
							print('%s is interacting with %s' % (LABELS[classIDs[i]],LABELS[classIDs[j]]))


		# show and write the output image
		cv2.imshow("Image", frame)
		k = cv2.waitKey(1)
		if k == 27:         # wait for ESC key to exit
			cv2.destroyAllWindows()
			break
		# elif k == ord('s'): # wait for 's' key to save and exit
		# 	cv2.imwrite('Outputs/'+args['input'].replace('.MOV','frame-%d-output.jpg' % frames).replace('Inputs/',''), frame)
		# 	cv2.destroyAllWindows()

	if writer is None:

		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# write the output frame to disk
	writer.write(frame)

print("[INFO] cleaning up...")
writer.release()
vs.release()
