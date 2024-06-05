import numpy as np # linear algebra
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
import cv2
import math
import os
from keras.models import load_model
import time
import threading
import mysql.connector
from mysql.connector import Error
import datetime
import io
import dropbox
from firebase import firebase
import datetime
import base64
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials, firestore, storage

predicted_result = ["def", "def", "def"]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="fire-detection-bae7c-309d1a472cf3.json"
# Init firebase with your credentials
cred = credentials.Certificate("fire-detection-bae7c-309d1a472cf3.json")
'''firestore authentication'''
firebase = firebase.FirebaseApplication('https://fire-detection-bae7c-default-rtdb.firebaseio.com/', None)  

db = firestore.Client()

IMG_SIZE = 224
NUM_EPOCHS = 20
NUM_CLASSES = 3
TRAIN_BATCH_SIZE = 77
TEST_BATCH_SIZE = 1 
loop_count = 0
def get_label_dict(train_generator ):
# Get label to class_id mapping
    labels = (train_generator.class_indices)
    label_dict = dict((v,k) for k,v in labels.items())
    return  label_dict   

def draw_prediction( frame, class_string ):
    x_start = frame.shape[1] -600
    cv2.putText(frame, class_string, (x_start, 75), cv2.FONT_HERSHEY_TRIPLEX, 2.5, (255, 0, 0), 7, cv2.LINE_AA)
    return frame

def prepare_image_for_prediction( img):
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def get_display_string(pred_class, label_dict):
    txt = ""
    for c, confidence in pred_class:
        txt += label_dict[c]
        if c :
            txt += '['+ str(confidence) +']'
    return txt

trained_model_l = load_model('model1.h5')
#trained_model_l = load_model('detection_model-ex-33--loss-4.97.h5')


data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                #sear_range=0.01,
                                zoom_range=[0.9, 1.25],
                                horizontal_flip=True,
                                vertical_flip=False,
                                data_format='channels_last',
                                brightness_range=[0.5, 1.5]
                               )
train_generator = data_generator_with_aug.flow_from_directory(
            'C:/Project/data/img_data/train',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=TRAIN_BATCH_SIZE,
            class_mode='categorical')

label_dict_l = get_label_dict(train_generator )

# Any results you write to the current directory are saved as output.
def predict(model, label_dict ):
    #url = "http://192.168.0.12:8080/"
    #vs = cv2.VideoCapture(url+"video")
    vs1 = cv2.VideoCapture(0)# use 1 insted of test1.mp4 to get frames from usb camera
    # vs2 = cv2.VideoCapture('test2.mp4')
    vs3 = cv2.VideoCapture('test3.mp4')
    fps = math.floor(vs1.get(cv2.CAP_PROP_FPS))
    ret_val = True
    writer = 0
    frame_count = 0
    while True:
        ret_val, frame1 = vs1.read()
        # ret_val, frame2 = vs2.read()
        ret_val, frame3 = vs3.read()
        if not ret_val:
            break
        resized_frame1 = cv2.resize(frame1, (IMG_SIZE, IMG_SIZE))
        frame_for_pred1 = prepare_image_for_prediction( resized_frame1 )
        pred_vec1 = model.predict(frame_for_pred1)
        # resized_frame2 = cv2.resize(frame2, (IMG_SIZE, IMG_SIZE))
        # frame_for_pred2 = prepare_image_for_prediction( resized_frame2 )
        # pred_vec2 = model.predict(frame_for_pred2)
        resized_frame3 = cv2.resize(frame3, (IMG_SIZE, IMG_SIZE))
        frame_for_pred3 = prepare_image_for_prediction( resized_frame3 )
        pred_vec3 = model.predict(frame_for_pred3)
        #print(pred_vec)
        pred_class1 =[]
        # pred_class2 =[]
        pred_class3 =[]
        confidence1 = np.round(pred_vec1.max(),2) 
        # confidence2 = np.round(pred_vec2.max(),2) 
        confidence3 = np.round(pred_vec3.max(),2) 

        if confidence1 > 0.4: 
            pc1 = pred_vec1.argmax()
            pred_class1.append( (pc1, confidence1) )
        else:
            pred_class1.append( (0, 0) )
        
        # if confidence2 > 0.4:
        #     pc2 = pred_vec2.argmax()
        #     pred_class2.append( (pc2, confidence2) )
        # else:
        #     pred_class2.append( (0, 0) )
        
        if confidence3 >0.4:
            pc3 = pred_vec3.argmax()
            pred_class3.append( (pc3, confidence3) )
        else:
            pred_class3.append( (0, 0) )
        
        if pred_class1:
            txt1 = get_display_string(pred_class1, label_dict)       
            frame1 = draw_prediction( frame1, txt1 )
            predicted_result[0] = txt1
        # if pred_class2:
        #     txt2 = get_display_string(pred_class2, label_dict)       
        #     frame2 = draw_prediction( frame2, txt2)
        #     predicted_result[1] = txt2
        if pred_class3:    
            txt3 = get_display_string(pred_class3, label_dict)       
            frame3 = draw_prediction( frame3, txt3 )
            predicted_result[2] = txt3

        if not writer:
            #fourcc = cv2.VideoWriter_fourcc(*"XVID")
            #writer = cv2.VideoWriter(filename, fourcc, fps,(frame.shape[1], frame.shape[0]), True)
            cv2.imwrite('Latest1.jpg',frame1)
            # cv2.imwrite('Latest2.jpg',frame2)
            cv2.imwrite('Latest3.jpg',frame3)    


        #frame_count +=1
        #fmpeg -sseof -3 -i test1_1.avi -update 1 -q:v 1 latest.jpg

        vs1.release()
        # vs2.release()
        vs3.release()
        #writer.release()  
   

#video_path = 'C:/Project/data/video_data/test_videos/test1.mp4'
def test():
    predict ( trained_model_l, label_dict_l)
    '''dropbox authentication'''

    access_token = 'your access token'
    file_from = ['Latest1.jpg','Latest2.jpg', 'Latest3.jpg']
    file_to = '/User1/'
    camera_list = ['Camera1', 'Camera2', 'Camera3']
    '''uploading to dropbox'''
    def upload_file(file_from, file_to):
        dbx = dropbox.Dropbox(access_token)
        f = open(file_from, 'rb')
        dbx.files_upload(f.read(), file_to, mode=dropbox.files.WriteMode.overwrite)
        result = dbx.files_get_temporary_link(file_to)
        #print(result.link)
        return result.link

    for index in range(len(file_from)):
        result = upload_file(file_from[index], str(file_to+file_from[index]))
        db.collection("User1").document(camera_list[index]).update({
            "Image": str(result),
            "Status": str(predicted_result[index]),
            db.field_path("Last updated"): firestore.SERVER_TIMESTAMP,
        })
        
    
while True:
    test()
time.sleep(180) #1 second



