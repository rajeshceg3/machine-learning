import numpy as np
import cv2
from image_demo_io import ROOT, options, model, finish
import time

args, image = options('eagle.jpg')

# Get labels from sysnet
slices = open(ROOT / 'imgClassifierModel_googlenet/synset_words.txt').read().strip().split("\n")
classes = [e[e.find(" ") + 1:].split(",")[0] for e in slices]

# Process the image to get a blob
Outputblob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# Load pretrained caffe model
net = model('imgClassifierModel_googlenet', 'bvlc_googlenet.prototxt.txt', 'bvlc_googlenet.caffemodel')

net.setInput(Outputblob)
preds = net.forward()

# Get top 2 prdictions
idxs = np.argsort(preds[0])[::-1][:2]

results = []
for (i, idx) in enumerate(idxs):
	results.append({'label': classes[idx], 'probability': float(preds[0][idx])})
	if i == 0: # First one is the most probable prediction
		text = "Label: {}, {:.2f}%".format(classes[idx],
			preds[0][idx] * 100)
		cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)


finish(args, image, results)
