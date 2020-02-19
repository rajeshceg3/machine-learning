import numpy as np
import cv2

# Fetch input image
image = cv2.imread("Lenna.png")
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# Load pre-trained model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Feed the blob to model
net.setInput(blob)
predictions = net.forward()
probability_threshold = 0.5 

for i in range(0, predictions.shape[2]):
	probability = predictions[0, 0, i, 2]
	if probability > probability_threshold:
		# Identify face location
		box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# Draw a box around face
		text = "{:.2f}%".format(probability * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Display the image window
cv2.imshow("Prediction", image)
cv2.waitKey(0)
