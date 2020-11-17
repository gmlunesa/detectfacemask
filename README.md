# detectfacemask

**detectfacemask** is a MobileNetV2-based machine learning model that aims to automatically detect faces in an image or live feed, and determine if the faces are wearing mask.

## Frameworks
* MobileNetV2 architecture
  * MobileNetV2 is an effective out-of-the-box feature extractor for object detection and segmentation. 
* Tensorflow and Keras
  * TensorFlow is an open source library for numerical computation and large-scale machine learning. Keras is a neural networks library written in Python that works as a wrapper for TensorFlow.
* BerkleyVision Caffe
  * Caffe is a deep learning framework made with expression, speed, and modularity.
* OpenCV
  * OpenCV is a library of programming functions mainly aimed at real-time computer vision.

## Features and Usage
### Face mask detection in images
Faces in image files are automatically detected, and an estimation of presence or absence of face mask is given. Press `ESC` to close the window.

On the project folder, run the following command on the terminal:  

```
python dfm-i.py --image <image file>
```

or

```
python dfm-i.py -i <image file>
```

Example:

```
python dfm-i.py --image test/subject1.jpg
```


### Face mask detection in live feed
Live feed through webcam will be enabled. A window will be launched that will show the webcam feed. Faces in the said feed are automatically detected, and an estimation of presence or absence of face mask is given. Press `ESC` to close the window.

On the project folder, run the following command on the terminal:  

```
python dfm-c.py
```

## Development
Clone the repository

```
$ git clone https://github.com/gmlunesa/detectfacemask.git
```

Change the directory to the cloned repo folder and create a new Python virtual environment

```
$ mkvirtualenv env-test
```

Run command to install required libraries

```
$ pip3 install -r requirements.txt
```

### Training

Please open `01. Data Preprocessing.ipynb` and `02. Training ML Model.ipnyb` for detailed explanations and code of the training process. This requires Jupyter Notebook.

## Dataset
Real World Masked Face Dataset ([RMFD](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)) by Baojin Huang [Arxiv](https://arxiv.org/abs/2003.09093)

## Demo
### Image detection
<img src='https://raw.githubusercontent.com/gmlunesa/detectfacemask/master/assets/dfm-i-demo-subject1.png' width="250px" alt="Jeff Mangum is NOT wearing a mask"/><img src='https://raw.githubusercontent.com/gmlunesa/detectfacemask/master/assets/dfm-i-demo-subject2.png' width="250px" alt="Du29+1 is wearing a mask I guess"/>

### Video detection
<img src='https://raw.githubusercontent.com/gmlunesa/detectfacemask/master/assets/dfc-i-demo-subject1.jpg' width="250px" alt="@gmlunesa not wearing a mask"/><img src='https://raw.githubusercontent.com/gmlunesa/detectfacemask/master/assets/dfc-i-demo-subject2.jpg' width="250px" alt="@gmlunesa wearing a mask"/>

## Inquiries

For any questions or clarifications, please see my contact details at [https://gmlunesa.com/contact](https://gmlunesa.com/contact).

For issues, please file them [here](https://github.com/gmlunesa/detectfacemask/issues/new).

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)

