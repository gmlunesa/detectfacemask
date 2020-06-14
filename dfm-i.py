# Import the necessary packages and libraries
import numpy as np
import argparse
import cv2
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Arguments are expected to be the --image <file name>
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
  help="Specify the path to image")
args = vars(ap.parse_args())

# Declare constants
faceDetectorPath = 'caffe_face_detector'
modelFile = 'mask_detector.model'
minConfidence = 0.5
imageSize = 224

boxLabelWithMask = 'Wearing mask'
boxLabelWithoutMask = 'Not wearing mask'

# Load Caffe-based face detector model file
prototxtPath = os.path.sep.join([faceDetectorPath, "deploy.prototxt"])
weightsPath = os.path.sep.join([faceDetectorPath,
  "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load mask detector model file generated from
# 02. Training the ML Model.ipynb
maskDetectorModel = load_model(modelFile)

# Load specified input image
image = cv2.imread(args["image"])
# Clone the input image
orig = image.copy()
# Retrieve image spatial dimensions
(h, w) = image.shape[:2]

print("[LOG] Image successfully retrieved.")

# Preprocess image:
  # Generate a blob from the input image
  # Scalefactor: 1.0
  # Size: 300x300
  # Mean subtraction: RGB (104, 177, 123)
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
  (104.0, 177.0, 123.0))

# Pass the generated image blob to the face detector network
faceNet.setInput(blob)

# Get face detections
detections = faceNet.forward()

# Loop over the face detections
for i in range(0, detections.shape[2]):

  # Obtain the confidence / probability associated with each detection
  confidence = detections[0, 0, i, 2]

  # Only do computations on detections with greater confidence
  # than set minimum
  if confidence > minConfidence:

    # Obtain x,y coordinates of the output bounding box
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    # Verify the bounding box is within the range of the
    # frame's dimensions
    (startX, startY) = (max(0, startX), max(0, startY))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

    # Get the region of interest- the face
    # Convert BGR channel to RGB channel
    # Resize and preprocess it
    try:
      face = image[startY:endY, startX:endX]
      face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
      face = cv2.resize(face, (imageSize, imageSize))
      face = img_to_array(face)
      face = preprocess_input(face)
      face = np.expand_dims(face, axis=0)

    except Exception as e:
      print(str(e))

    # Pass the processed facial region of interest to the
    # mask detector network model
    (withMask, withoutMask) = maskDetectorModel.predict(face)[0]

    # Assign class label
    label = "With Mask" if withMask > withoutMask else "No Mask"

    # Configure display text and color
    if label == "With Mask":
      boxLabel = boxLabelWithMask
      color = (50, 205, 50)
    
    else:
      boxLabel = boxLabelWithoutMask
      color =  (50, 50, 205)

    # Insert probability to the display text
    boxLabel = "{}: {:.2f}%".format(boxLabel, max(withMask, withoutMask) * 100)

    # Show display text and bounding box on the output window
    cv2.putText(image, boxLabel, (startX, startY - 10),
      cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Show display text and bounding box on the output window
cv2.imshow("Output", image)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output", 600, 600)

cv2.waitKey(0)