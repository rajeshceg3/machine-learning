import cv2
import numpy as np

# SSD is used along with mobilenet neural network. 
# Following are the ssd classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# Load pretrained neural network from Caffee
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Process the image to get blob output
image = cv2.imread("bookshelf_cat.jpg")
(h, w) = image.shape[:2]
OutputBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# Feed the blob to neural network
net.setInput(OutputBlob)

#Get the predictions fron network
predictions = net.forward()

probability_threshold = 0.3
for i in np.arange(0, predictions.shape[2]):
	probability = predictions[0, 0, i, 2]

	# thresholding
	if probability > probability_threshold:
		# Index of class label
		index = int(predictions[0, 0, i, 1])
		# Location of object
		box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# Display the results
		label = "{}: {:.2f}%".format(CLASSES[index], probability * 100)
		cv2.rectangle(image, (startX, startY), (endX, endY),(0,255,0), 2)
		cv2.putText(image, label, (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Display the image
cv2.imshow("Result", image)
cv2.waitKey(0)
