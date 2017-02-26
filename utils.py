import PIL.Image
from cStringIO import StringIO
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import os

import cv2

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from IPython.display import Image, display


def get_image_from_path(img_path):
    return image.load_img(img_path, target_size=(224, 224))


def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a.astype('uint8')).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))
    
    
def identify_picture(model, img):
    x = np.expand_dims(image.img_to_array(img), 0)
    x = preprocess_input(x)
    
    display(img)
    showarray(x[0])

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    preds = decode_predictions(preds, top=3)[0]

    for i in preds:
        print 'Prediction: {}\tLikelihood: {:0.1f}%'.format(i[1], i[2]*100)
        
    return preds[0]


def get_webcam_img():
    while True:
        cam = cv2.VideoCapture(0)
        s, img = cam.read()
        if s:
            cv2.imwrite("tmp.jpg",img)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sleep(1)
        
        
def get_webcam_img():
    while True:
        cam = cv2.VideoCapture(0)
        s, img = cam.read()
        if s:
            cv2.imwrite("tmp.jpg",img)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sleep(1)
        
        
def what_is_in_front_of_you(model):
    webcam_img = get_webcam_img()
    webcam_img = identify_picture(model, get_image_from_path('tmp.jpg'))

    os.system("say 'I think I am seeing a {}.  I am {:0.1f} percent sure' &".format(webcam_img[1], webcam_img[2]*100))