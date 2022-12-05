#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:56:52 2022

@author: shivam
"""
import torch
import cv2
import glob
#import logging as lg
import time
import os
from datetime import datetime

#logger = lg.basicConfig(filename = 'D:\\tests.log', level = lg.INFO, format = '%(asctime)s %(message)s')     
#logger = lg.getLogger()
dest_img_path = '/app/output'
model = torch.hub.load('./yolov5', 'custom', path='./yolov5/ppe_best.pt',source='local',force_reload=True)

path = glob.glob('/app/images/images/*')
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
print("for loop started")
for file in path:
    img = cv2.imread(file)
    #img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
    print("image Received")
    time_infer1 = time.time()
    results1 = model(img, size =640)
    results = results1.pandas().xyxy[0].to_dict(orient="records")
    for result in results: 
        con = result['confidence']
        cs = result['name']
        #if cs == 'person':  #and con > 0.4:
        print('image saved',count)
        draw_bboxes(img, results)
        t = time.time()
        dt = datetime.now()
        imgname = 'img_'+str(dt.date())+"_"+str(dt.hour)+\
                '-'+str(dt.minute)+'-'+str(dt.second)+'_'+\
                str(dt.microsecond//1000)
        imgname += '_iocl.jpg'
        cv2.imwrite(os.path.join(dest_img_path, imgname), img)
        break
    count += 1 
