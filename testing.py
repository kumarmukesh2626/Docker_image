#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:40:09 2022

@author: shivam
"""


import torch
import cv2
import glob
#import logging as lg
import time
import os
from datetime import datetime



dest_img_path = './output'
model = torch.hub.load('./yolov5', 'custom', path='./yolov5/crowdhuman_yolov5m.pt',source='local',force_reload=True)

path = glob.glob('./images/*')
count=1
time_infer = time.time()
print("Model Loaded")
def draw_bboxes(img, results_dict):
    '''
    Parameters
    ----------
    img : np.ndarray
        uint8 images grabbed by camera
    results_dict : dict
        results dictionary with all the defects detected, and corresponding
        bboxes coordinates and confidence scores

    Returns
    -------
    img : np.ndarray 
        With bboxes drawn on the image
    '''
    def_count = 0
    max_display_defects = 4
    for result in results_dict:
        con = result['confidence']
        cs = result['name']
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)
        text = "{}: {:.2f}".format(cs, con)
        cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 2)
        def_count += 1 
        # Will display at max max_display_defects on the image
        if def_count == max_display_defects:
            break
    return




cap = cv2.VideoCapture('rtsp://admin:Admin@123!@103.210.28.115:1070/live2.sdp')
frame_id = 0

while(cap.isOpened()):
    time_infer = time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    count+=1
    frame_id +=1
    
    if ret == True:
         if frame_id% 50 == 0:
             dt = datetime.now()
             imgname = 'img_'+str(dt.date())+"_"+str(dt.hour)+\
                                '-'+str(dt.minute)+'-'+str(dt.second)+'_'+\
                                str(dt.microsecond//1000)
             imgname += '_sunbeam.jpg'
             img = 'img_'+str(frame_id)
             img += '_sunbeam.jpg'
             print(ret,"frame_id {}".format(frame_id))
             results1 = model(frame,size = 640)
             # crops = results1.crop(save=False) 
             # logger.info({"Model Inference Time :":time.time()-start})
             class_ids = []
             confidences = []
             boxes = []
             results = results1.pandas().xyxy[0].to_dict(orient="records")

             for result in results:                  
                 confidence = result['confidence']
                 label = result['name']
                 class_id = result['class']
                 if label == 'person' and confidence >= 0.25:
                     
                     x1 = int(result['xmin'])
                     y1 = int(result['ymin'])
                     x2 = int(result['xmax'])
                     y2 = int(result['ymax'])
                    #img1 = img.copy()
                     boxes.append([x1, y1, x2, y2])
                     confidences.append(float(confidence))
                     class_ids.append(class_id)
                     print("class {} : confidence {} ".format(label,confidence)) 
                     
cap.release()
             #cv2.imwrite(os.path.join(img_path, img), frame)