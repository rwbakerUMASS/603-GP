#!/usr/bin/env python3
import rospy
import numpy as np
import time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PoseArray, Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
import math
import cv2
import os
import random
from scipy import ndimage
import tf
import pickle
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image as PIL_Image
from sensor_msgs.msg import Image as sensor_msgs_Image
from ultralytics.utils.plotting import Annotator
from cv_bridge import CvBridge
import message_filters

class Image_Processor:
    def __init__(self) -> None:
        self.yolo_model = YOLO("yolov9c.pt").to('cuda')
        self.cv2_bridge = CvBridge()
        self.def_width = 640
        self.def_height = 480
        self.centering_threshold = 0.1
        self.img_center = (self.def_width/2, self.def_height/2)
        self.depth_img_data = None
        self.mask = None
        self.FOV = 87
        self.angle = 0
        self.new_frame = False
        self.human_found = False

    def is_new_frame(self):
        return self.new_frame

    def get_angle(self):
        self.new_frame = False
        return self.angle

    def check_box_math(self, box_center, b):
        fig, ax = plt.subplots(figsize=(8, 6))  # figsize is specified in inches
        # Plot the point (100, 100)
        ax.plot(box_center[0], box_center[1], 'ro')  # 'ro' specifies red color ('r') and circle marker ('o')
        ax.plot(self.img_center[0], self.img_center[1], 'go')
        ax.plot(b[0], self.def_height-b[1], 'yo')
        ax.plot(b[2], b[3], 'bo')
        ax.set_xlim(0, self.def_height)
        ax.set_ylim(0, self.def_width)
        # Display the plot
        plt.show()

    def detect_human(self, rgb_img:sensor_msgs_Image, depth_img:sensor_msgs_Image):
        # print(f'RGB Encoding: {rgb_img.encoding}')
        cv_image = self.cv2_bridge.imgmsg_to_cv2(rgb_img, desired_encoding=rgb_img.encoding)
        results = self.yolo_model.predict(source=cv_image, save=True)
            # cv2.imshow('x',cv_image)
            # cv2.waitKey(1)

        boxes = results[0].boxes # to check: results should be just 1 element when 1 img is passed to model
        self.human_found = False
        for i, box in enumerate(boxes):
            if box.cls[0] == 0: # if box bounds a person
                self.human_found = True
                # self.mask = np.array(results[0].masks[i].data[0].detach().numpy(),dtype='bool')
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                print(box.cls[0], box.conf, b) # print object type (person), prediction confidence, coords
                box_center = (b[0] + int((b[2]-b[0])/2), self.def_height-b[1] + int((b[3]-self.def_height-b[1])/2))
                self.box = box_center
                # self.check_box_math(box_center, b)

                self.angle = ((box_center[0] - self.img_center[0])/self.def_width)*self.FOV
                print(f'ANGLE: {self.angle}')
                self.new_frame = True


                # Check if the bounding box center is at the center of the image
                if abs(box_center[0] - self.img_center[0]) < self.def_width * self.centering_threshold:
                    print("Bounding box center is near the center of the image")
                else:
                    print("Bounding box center is not at the center of the image")
        if not self.human_found:
            self.angle = 0
                    # orient robot to have human in the center using depth info
        # print(f'Depth Encoding: {depth_img.encoding}')
        # self.depth_img_data = depth_img.data
        # try:
        #     cv_image = self.cv2_bridge.imgmsg_to_cv2(depth_img, desired_encoding=depth_img.encoding)
        #     print(np.mean(cv_image[self.mask]))
        # except Exception as e:
        #     print(f"depth_img_callback: {e}")

    # def detect_human(self, rgb_img:sensor_msgs_Image):
    #     print(f'RGB Encoding: {rgb_img.encoding}')
    #     cv_image = self.cv2_bridge.imgmsg_to_cv2(rgb_img, desired_encoding=rgb_img.encoding)
    #     results = self.yolo_model.predict(source=cv_image, save=True)
    #         # cv2.imshow('x',cv_image)
    #         # cv2.waitKey(1)

    #     boxes = results[0].boxes # to check: results should be just 1 element when 1 img is passed to model
    #     for i, box in enumerate(boxes):
    #         if box.cls[0] == 0: # if box bounds a person
    #             self.mask = np.array(results[0].masks[i].data[0].detach().numpy(),dtype='bool')
    #             b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
    #             print(box.cls[0], box.conf, b) # print object type (person), prediction confidence, coords
    #             box_center = (b[0] + int((b[2]-b[0])/2), self.def_height-b[1] + int((b[3]-self.def_height-b[1])/2))
    #             self.box = box_center
    #             # self.check_box_math(box_center, b)

    #             # Check if the bounding box center is at the center of the image
    #             if abs(box_center[0] - self.img_center[0]) < self.def_width * self.centering_threshold:
    #                 print("Bounding box center is near the center of the image")
    #             else:
    #                 print("Bounding box center is not at the center of the image")
    #                 # orient robot to have human in the center using depth info
                
    # def depth_img_callback(self, depth_img:sensor_msgs_Image):
    #     print(f'Depth Encoding: {depth_img.encoding}')
    #     self.depth_img_data = depth_img.data
    #     try:
    #         cv_image = self.cv2_bridge.imgmsg_to_cv2(depth_img, desired_encoding=depth_img.encoding)
    #         # cv2.imshow('x',cv_image)
    #         # cv2.waitKey(1)
    #     except Exception as e:
    #         print(f"depth_img_callback: {e}")

class Camera:
    def __init__(self, rate) -> None:
        self.img_processor = Image_Processor()
        self.rgb_img_sub = message_filters.Subscriber("/camera/color/image_raw", sensor_msgs_Image)
        self.depth_img_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", sensor_msgs_Image)
        self.ts = message_filters.TimeSynchronizer([self.rgb_img_sub,self.depth_img_sub],10)
        self.ts.registerCallback(self.img_processor.detect_human)
        
        self.rate = rate
        # rospy.spin() 

    def is_new_frame(self):
        return self.img_processor.is_new_frame()

    def get_angle(self):
        return self.img_processor.get_angle()

if __name__ == '__main__':
    try:
        # time.sleep(5) # wait a bit to ensure the Gazebo GUI is visible before running
        rospy.init_node("camera")
        rate = rospy.Rate(10)
        cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        camera = Camera(rate)
        
        angle = 0

        while True:
            angle = camera.get_angle()

            angle = math.radians(angle)
            vel = Twist()
            vel.angular.z = -2 * np.power(angle,3)
            cmd_vel.publish(vel)
            rate.sleep()
    except Exception as e:
        print(e)








