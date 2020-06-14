# Import the necessary packages and libraries
import numpy as np
import imutils
import time
import cv2
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

def detect_and_predict_mask(frame, faceNet, maskDetectorModel):

  # Get the frame's dimensions
  (h, w) = frame.shape[:2]

  # Generate a blob from the input frame feed
    # Scalefactor: 1.0
    # Size: 300x300
    # Mean subtraction: RGB (104, 177, 123)
  blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
    (104.0, 177.0, 123.0))

  # Pass the generated image blob to the face detector network
  faceNet.setInput(blob)

  # Get face detections
  detections = faceNet.forward()

  # Initialize the following:
    # List of faces
    # List of the corresponding locations
    # List of predictions from the mask detector model
  faces = []
  locations = []
  predictions = []

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
        face = frame[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (imageSize, imageSize))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

      except Exception as e:
        print(str(e))

      # Add the face to the faces list
      faces.append(face)
      # Add its location (bounding boxes) to the locations list
      locations.append((startX, startY, endX, endY))

  # Make predictions if there is at least one face detected
  if len(faces) > 0:
    # Pass the processed facial region of interest to the
    # mask detector network model
    predictions = maskDetectorModel.predict(faces)

  # Return the locations / bounding boxes and predictions for the faces
  return (locations, predictions)

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

# Open camera feed
print("[LOG] Opening camera feed.")
vs = VideoStream(src=0).start()
# Insert a 2 second buffer for initialization 
time.sleep(2.0)

# Loop over the frames for the camera stream
while True:

  # Get frame from the camera video stream
  # Set maximum width of the frame to 600
  frame = vs.read()
  frame = imutils.resize(frame, width=600)

  # detect faces in the frame and determine if they are wearing a
  # face mask or not

  # Call defined function to detect faces and
  # predict presence of mask
  (locations, predictions) = detect_and_predict_mask(frame, faceNet, maskDetectorModel)

  # Loop over the detected bounding boxes for the faces
  for (box, prediction) in zip(locations, predictions):
    # unpack the bounding box and predictions

    # Assign bounding box dimensions
    (startX, startY, endX, endY) = box

    # Retrieve mask predictions
    (withMask, withoutMask) = prediction

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
    cv2.putText(frame, boxLabel, (startX, startY - 10),
      cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

  # Show display text and bounding box on the output window
  cv2.imshow("Output", frame)
  key = cv2.waitKey(1) & 0xFF

  # Exit on Esc key press
  if key == 27:
    break

# Terminate all programs
cv2.destroyAllWindows()
vs.stop()