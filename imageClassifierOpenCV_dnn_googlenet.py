import numpy as np
import cv2
import time

image = cv2.imread("eagle.jpg")

# Get labels from sysnet
slices = open("imgClassifierModel_googlenet/synset_words.txt").read().strip().split("\n")
classes = [e[e.find(" ") + 1:].split(",")[0] for e in slices]

# Process the image to get a blob
Outputblob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# Load pretrained caffe model
net = cv2.dnn.readNetFromCaffe("imgClassifierModel_googlenet/bvlc_googlenet.prototxt", "imgClassifierModel_googlenet/bvlc_googlenet.caffemodel")

net.setInput(Outputblob)
preds = net.forward()

# Get top 2 prdictions
idxs = np.argsort(preds[0])[::-1][:2]

for (i, idx) in enumerate(idxs):
	if i == 0: # First one is the most probable prediction
		text = "Label: {}, {:.2f}%".format(classes[idx],
			preds[0][idx] * 100)
		cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)

	# Post the prediction
	print("Potential label {} with Probability: {:.5}".format(i + 1,classes[idx], preds[0][idx]))

cv2.imshow("Image",image)
cv2.waitKey(0)
