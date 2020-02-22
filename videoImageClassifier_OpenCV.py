from imutils.video import FileVideoStream
import numpy as np
import imutils
import cv2

# SSD is used along with mobilenet neural network. 
# Following are the ssd classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# Assign random color choices
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load pretrained Caffee neural network 
net = cv2.dnn.readNetFromCaffe("imgClassifierModel_mobilenet/deploy.prototxt", "imgClassifierModel_mobilenet/mobilenet_iter_73000.caffemodel")

# Initialize video stream
vs = FileVideoStream("video.mp4").start()

probability_treshold = 0.76

# Process video frames one by one
while True:
	frame = vs.read()

	# Save video frame as a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# Feed the blob to neural network
	net.setInput(blob)
	predictions = net.forward()

	# Process  predictions iteratively
	for i in np.arange(0, predictions.shape[2]):
		# Calculate probability
		probability = predictions[0, 0, i, 2]
		if probability > probability_treshold:
			# Index of class label
			index = int(predictions[0, 0, i, 1])
			# Location of object
			box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Display the results
			label = "{}: {:.2f}%".format(CLASSES[index],probability * 100)
			cv2.rectangle(frame,(startX, startY),(endX, endY),COLORS[index], 2)			
			cv2.putText(frame, label,(startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 
			0.5, COLORS[index], 2)

	# Display the image frame
	cv2.imshow("Video Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# Activities to do on destruction
cv2.destroyAllWindows()
vs.stop()
