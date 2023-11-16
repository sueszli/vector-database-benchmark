"""
TSRS - Detect Road Signs with Live Video
Rameet Sekhon - rameet.sekhon@uoit.net
Thagshan Mohanarathnam
"""

import numpy as np
import argparse as ap
import time
import cv2
import os
import math
import itertools

"""
This path should be specified through the command line to the path
where the darknet binaries reside. 
Darknet can be obtained from: https://github.com/AlexeyAB/darknet
"""
DARKNET_PATH = "D:/ComputerVision/finalproject/darknet/build/darknet/x64"

"""
The path to the config, label and weight files relative to the darknet 
binary path.
"""
DARKNET_LABEL_PATH = "./cfg/obj.names"
DARKNET_CONFIG_PATH = "./cfg/yolo-obj.cfg"
DARKNET_WEIGHTS_PATH = "./backup/yolo-obj_60000.weights"

"""
Program entrypoint
"""
if __name__ == "__main__":
	# Parse arguments
	argParser = ap.ArgumentParser()
	argParser.add_argument("-d", "--darknet", required=False, type=str, default=DARKNET_PATH,
		help="path to darknet binaries")
	argParser.add_argument("-v", "--video", required=False, type=str, default="",
		help="path to video")
	argParser.add_argument("-s", "--skip", required=False, type=int, default=-1, 
		help="after how many frames do you want it to skip a frame")
	argParser.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum confidence to remove bad detections")
	argParser.add_argument("-t", "--threshold", type=float, default=0.2,
		help="threshold when applying non-maxima suppression")
	args = argParser.parse_args()

	# Read the classes
	classes = open(os.path.join(args.darknet, DARKNET_LABEL_PATH)).read().strip().split("\n")

	# Create random colors for each of the classes
	np.random.seed(int(time.time()))
	colors = [ (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for c in classes ]

	# Load our YOLO dataset trained on road signs (10 classes)
	print("Loading YOLO dataset from disk...")
	net = cv2.dnn.readNetFromDarknet(os.path.join(args.darknet, DARKNET_CONFIG_PATH), os.path.join(args.darknet, DARKNET_WEIGHTS_PATH))

	# Set preferred device as any available OpenCL/CUDA device
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

	# Create video capture depending on argument supplied (use webcam if no path was specified)
	print("Video specified:", args.video)
	cap = cv2.VideoCapture(0) if args.video == "" else cv2.VideoCapture(args.video)

	# Initialize frame skipping value
	frameCount = 0

	print("Running detection")

	# Perform detection on video
	while cap.isOpened():
		# Read a frame
		ret, image = cap.read()

		# Check if we should skip the frame
		if args.skip != -1 and frameCount == args.skip:
			frameCount = 0
			continue

		# Increment frame count
		frameCount += 1

		# Get the image dimensions
		imgHeight, imgWidth = image.shape[:2]
		
		# Determine only the output layer names that we need from darknet
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		# Preprocess the image in order to normalize them using mean subtraction, scaling
		# and swapping the red and blue color channels, then convert to a binary data type
		# to be used for computation.
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)

		# Perform a forward pass of the darknet YOLO object detector, and obtain the 
		# bounding boundingBoxes for the classes, and their associated confidence
		net.setInput(blob)

		start = time.time()
		outputLayers = net.forward(ln)
		end = time.time()
		
		# Show timing information for how long the operation took
		#print("YOLO took {:.6f} seconds".format(end - start))

		# Initialize our list of results
		results = list()

		# Check all of the outputs for the layers
		for output in outputLayers:
			# Loop over each of the detections
			for detection in output:
				# Get the properties of the detected object
				scale = detection[0:4]
				scores = detection[5:]
				classIndex = np.argmax(scores)
				confidence = scores[classIndex]

				# Ensure the confidence is greater than the specified threshold
				if confidence > args.confidence:
					# Since the YOLO detector scales down images internally to make it easier
					# to process, it also returns the scales that it scaled down the original
					# image using, so we can restore the size of the bounding box
					box = (scale * np.array([ imgWidth, imgHeight, imgWidth, imgHeight ])).astype("int")

					# Update the box with the correct coordinates, since YOLO uses a center x/y 
					# coordinate system
					box[0] = int(box[0] - (box[2] / 2))
					box[1] = int(box[1] - (box[3] / 2))

					# Store results
					results.append((box, float(confidence), classIndex))

		# Apply a non-maxima suppression to remove overlapping bounding boxes
		bb = [ r[0] for r in results ] # Bounding boxes
		sc = [ r[1] for r in results ] # Scores
		indexes = cv2.dnn.NMSBoxes(bb, sc, args.confidence, args.threshold)

		# Draw bounding box and label over each detection
		for i in itertools.chain.from_iterable(indexes):
			# Get the result
			r = results[i]
			x, y, w, h = r[0] # Bounding box
			c = r[1] # Confidence
			classIndex = r[2] # Class index
			className = classes[classIndex] # Class name

			# Draw a rectangle around the box and a label above it
			color = tuple(colors[classIndex])
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			cv2.putText(image, "{}: {:.2f}".format(className, c), (x, y - 6), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
		
		# Show the output image
		cv2.imshow("Image", image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release handles
	cap.release()
	cv2.destroyAllWindows()