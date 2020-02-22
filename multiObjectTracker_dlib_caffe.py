from imutils.video import FileVideoStream
import numpy as np
import imutils
import cv2
import dlib

# MobileNet Classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# Load pretrained neural network from Caffee
net = cv2.dnn.readNetFromCaffe("objTrackerModel\deploy.prototxt","objTrackerModel\mobilenet_iter_73000.caffemodel")
vs = FileVideoStream("multiObject.mp4").start()

# Initialize empty list for trackers and labels
trackers = []
labels = []

probability_threshold = 0.95
objectFound = False
# Process video frames iteratively
while True:

	frame = vs.read()
	# Maintain aspect ratio while resizing
	frame = imutils.resize(frame, width=600)
	frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

	if objectFound is False:
		# Load blob from Image
		(height, width) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame,0.007843, (width, height), 127.5)

		# Feed the blob to model
		net.setInput(blob)
		predictions = net.forward()

		# loop over the predictions
		for i in np.arange(0, predictions.shape[2]):
			probability = predictions[0, 0, i, 2]
			if probability > probability_threshold:
				objectFound = True
				# Identify of index of detected class
				index = int(predictions[0, 0, i, 1])
				labels.append(CLASSES[index])
				# Identify of location of object
				box = predictions[0, 0, i, 3:7] * np.array([width, height, width, height])
				(startX, startY, endX, endY) = box.astype("int")

				# Tracking operation
				tracker = dlib.correlation_tracker()
				track_area = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(frame_RGB, track_area)

				# Add tracker to the list
				trackers.append(tracker)

				# Display the result
				label = "{}: {:.2f}%".format(CLASSES[index],probability * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0,255,0), 2)			
				cv2.putText(frame, label, (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

	else:

		# Iterate for all trackers
		for tracker, label in zip(trackers,labels):
			# Get position for new frame
			tracker.update(frame_RGB)
			new_loc = tracker.get_position()
			startX = int(new_loc.left())
			startY = int(new_loc.top())
			endX = int(new_loc.right())
			endY = int(new_loc.bottom())

			# Update the bounding box with new location
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
			cv2.putText(frame, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# Display output
	cv2.imshow("Video Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# Some housekeeping
cv2.destroyAllWindows()
vs.stop()
