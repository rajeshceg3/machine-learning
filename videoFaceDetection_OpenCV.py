from imutils.video import FileVideoStream
import numpy as np
import imutils
import cv2

# Load pretrained neural network from Caffee
net = cv2.dnn.readNetFromCaffe("faceDetectionModel_ssd\deploy.prototxt", "faceDetectionModel_ssd\res10_300x300_ssd_iter_140000.caffemodel")

vs = FileVideoStream("People.mp4").start()
probability_threshold = 0.5

# Process video frames
while True:

	frame = vs.read()
	frame = cv2.resize(frame, (300,300))
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# Feed blob to model
	net.setInput(blob)
	predictions = net.forward()

	# Process predictions iteratively
	for i in range(0, predictions.shape[2]):

		probability = predictions[0, 0, i, 2]
		if probability > probability_threshold:
			# Identify location of people
			box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	 
			# Display identified location as a rectangle
			text = "{:.2f}%".format(probability * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# Display the output
	cv2.imshow("Video Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# A little bit of housekeeping
cv2.destroyAllWindows()
vs.stop()
