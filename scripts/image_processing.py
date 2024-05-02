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
from scipy import ndimage, stats
import tf
import pickle
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image as PIL_Image
from sensor_msgs.msg import Image as sensor_msgs_Image
# from sensor_msgs.msg import CompressedImage as sensor_msgs_Image
from ultralytics.utils.plotting import Annotator
from cv_bridge import CvBridge
import message_filters
import pyrealsense2 as rs

class Image_Processor:
    def __init__(self) -> None:
        self.yolo_model = YOLO("yolov9c-seg.pt").to('cuda')
        self.cv2_bridge = CvBridge()
        self.def_width = 640
        self.def_height = 480
        self.centering_offset_factor = 0.035
        self.centering_threshold = self.def_width * self.centering_offset_factor
        self.img_center = (self.def_width/2, self.def_height/2)
        self.depth_img_data = None
        self.mask = None
        self.FOV = 87
        self.angle = 0 # radians
        self.dist = None
        self.human_detected = False
        self.yolo_pub = rospy.Publisher('/yolo',sensor_msgs_Image)
        # self.dist_thresh = 200

        self.rgb_img = None
        self.depth_img = None

    def get_dist(self) -> float:
        return self.dist

    def get_angle(self) -> float:
        return self.angle

    def check_box_math(self, box_center, b) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))  # figsize is specified in inches
        ax.plot(box_center[0], box_center[1], 'ro')  # 'ro' specifies red color ('r') and circle marker ('o')
        ax.plot(self.img_center[0], self.img_center[1], 'go')
        ax.plot(b[0], self.def_height-b[1], 'yo')
        ax.plot(b[2], b[3], 'bo')
        ax.set_xlim(0, self.def_height)
        ax.set_ylim(0, self.def_width)
        plt.show()

    def get_newest_frame(self, rgb_img:sensor_msgs_Image, depth_img:sensor_msgs_Image = None):
        # if self.rgb_img is not None:
        #     print('GOT NEW COLOR FRAME {:3f}'.format((self.depth_img.header.stamp.to_time()-self.rgb_img.header.stamp.to_time())))
        #     print('GOT NEW DEPTH FRAME {:3f}'.format((depth_img.header.stamp.to_time()-self.depth_img.header.stamp.to_time())))
        self.rgb_img = rgb_img
        self.depth_img = depth_img

    # def get_newest_depth_frame(self, depth_img:sensor_msgs_Image):
    #     if self.depth_img is not None:
    #         print('GOT NEW DEPTH FRAME {:3f}'.format((depth_img.header.stamp.to_time()-self.depth_img.header.stamp.to_time())))
    #     self.depth_img = depth_img

    def detect_human(self) -> None:
        if self.rgb_img is None or self.depth_img is None:
            return
        depth_img = self.depth_img
        rgb_img = self.rgb_img
        depth_cv_image = self.cv2_bridge.imgmsg_to_cv2(depth_img, desired_encoding=depth_img.encoding)
        cv_image = self.cv2_bridge.imgmsg_to_cv2(rgb_img, desired_encoding=rgb_img.encoding)
        results = self.yolo_model.predict(source=cv_image,save=True)
        yolo_img = cv2.imread(f'{results[0].save_dir}/image0.jpg')
        cv2.imshow('x',yolo_img)
        cv2.waitKey(1)
        boxes = results[0].boxes # to check: results should be just 1 element when 1 img is passed to model
        self.human_detected = False
        for i, box in enumerate(boxes):
            if box.cls[0] == 0: # if box bounds a person
                self.human_detected = True
                self.mask = np.array(results[0].masks[i].data[0].cpu().detach().numpy(),dtype='bool')
                depth_mode = stats.mode(depth_cv_image[self.mask][np.nonzero(depth_cv_image[self.mask])]).mode[0]
                # if not self.dist or abs(depth_mode-self.dist) < self.dist_thresh:
                self.dist = depth_mode
                # depth_mean = np.mean(np.nonzero(depth_cv_image[self.mask]))
                # print("DEPTH MEAN= {:3f}".format(depth_mean))
                # print("DEPTH MODE= {:3f}".format(depth_mode))
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                box_center = (b[0] + int((b[2]-b[0])/2), self.def_height-b[1] + int((b[3]-self.def_height-b[1])/2))
                self.angle = math.radians(((box_center[0] - self.img_center[0])/self.def_width)*self.FOV)
                print(f'Person detected with {box.conf.item()} confidence at {self.angle} radians, {depth_mode} mm away')
                break
        if not self.human_detected:
            self.angle = 0
            self.dist = None
 
    def depth_img_callback(self, depth_img:sensor_msgs_Image):
        print(f'Depth Encoding: {depth_img.encoding}')
        self.depth_img_data = depth_img.data
        try:
            cv_image = self.cv2_bridge.imgmsg_to_cv2(depth_img, desired_encoding=depth_img.encoding)
            # cv2.imshow('x',cv_image)
            # cv2.waitKey(1)
        except Exception as e:
            print(f"depth_img_callback: {e}")


class Camera:
    def __init__(self, rate) -> None:
        self.img_processor = Image_Processor()
        self.rgb_img_sub = message_filters.Subscriber("/camera/color/image_raw", sensor_msgs_Image)
        self.depth_img_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", sensor_msgs_Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_img_sub,self.depth_img_sub],1,0.1)
        self.ts.registerCallback(self.img_processor.get_newest_frame)
        # self.rgb_img_sub = rospy.Subscriber("/camera/color/image_raw", sensor_msgs_Image, self.img_processor.get_newest_color_frame)
        # self.rgb_depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw, sensor_msgs_Image, self.img_processor.get_newest_depth_frame)
        self.rate = rate

        rospy.wait_for_message("/camera/color/image_raw", sensor_msgs_Image)

    def process(self):
        self.img_processor.detect_human()

    def get_angle(self):
        return self.img_processor.get_angle()
    
    def get_dist(self):
        return self.img_processor.get_dist()


class Triton:
    def __init__(self, camera:Camera, rate) -> None:
        self.camera = camera
        self.rate = rate
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.vel_cmd = Twist()
    
    def move_to_human(self, goal_dist=1000, max_speed=0.5, min_speed=-0.3) -> None:

        angle = self.camera.get_angle()
        self.vel_cmd.angular.z = -0.5 * angle

        if not self.camera.get_dist():
            self.vel_pub.publish(Twist())
            return

        dist = self.camera.get_dist() - goal_dist
        self.vel_cmd.linear.x = max(min(0.5 * (dist/1000),max_speed),min_speed)

    def stop(self):
        self.vel_cmd = Twist()
        self.vel_pub.publish(self.vel_cmd)

    def run(self) -> None:
        while not rospy.is_shutdown():
            self.camera.process()
            if self.camera.img_processor.human_detected:
                self.move_to_human()
                self.vel_pub.publish(self.vel_cmd)
            else:
                self.stop()
            self.rate.sleep()


if __name__ == '__main__':
    try:
        # time.sleep(5) # wait a bit to ensure the Gazebo GUI is visible before running
        rospy.init_node("image_processing")
        rate = rospy.Rate(10)
        camera = Camera(rate)
        triton = Triton(camera, rate)
        triton.run()
    except Exception as e:
        print(e)