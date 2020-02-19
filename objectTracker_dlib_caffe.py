from imutils.video import FileVideoStream
import numpy as np
import imutils
import cv2
import dlib

# MobileNet classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# Load pretrained Caffee model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt","mobilenet_iter_73000.caffemodel")


vs = FileVideoStream("Video05.mp4").start()

# Create dlib_tracker with dlib
dlib_tracker = dlib.correlation_dlib_tracker()

probability_threshold = 0.95
objectFound = False
# Process video frames
while True:

	frame = vs.read()
	# Resize while maintaining aspect ratio
	frame = imutils.resize(frame, width=600)
	frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

	if objectFound is False:
		# convert to blob
		(height, width) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame,0.007843, (width, height), 127.5)

		# Feed the blob to model
		net.setInput(blob)
		predictions = net.forward()

		# Process predictions
		for i in np.arange(0, predictions.shape[2]):
			probability = predictions[0, 0, i, 2]
			if probability > probability_threshold:
				objectFound = True
				# Identify the index
				index = int(predictions[0, 0, i, 1])
				# Get the box location for drawing rectangle
				box = predictions[0, 0, i, 3:7] * np.array([width, height, width, height])
				(startX, startY, endX, endY) = box.astype("int")

				# Initiate tracking
				track_area = dlib.rectangle(startX, startY, endX, endY)
				dlib_tracker.start_track(frame_RGB, track_area)

				# Show the output
				label = "{}: {:.2f}%".format(CLASSES[index],probability * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0,255,0), 2)
				cv2.putText(frame, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
				break

	else:
		# Update existing dlib_tracker with current data

		dlib_tracker.update(frame_RGB)
		new_loc = dlib_tracker.get_position()

		startX = int(new_loc.left())
		startY = int(new_loc.top())
		endX = int(new_loc.right())
		endY = int(new_loc.bottom())

		# Update bounding box used for tracking
		cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
		cv2.putText(frame, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# Display the output
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# A little housekeeping
cv2.destroyAllWindows()
vs.stop()
