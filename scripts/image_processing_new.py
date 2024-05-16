#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist
import math
import cv2
from scipy import stats
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sensor_msgs.msg import Image as sensor_msgs_Image
from cv_bridge import CvBridge
import message_filters
import lidar_processing as lp
import time
import pickle

class Image_Processor:
    def __init__(self) -> None:
        self.yolo_model = YOLO("yolov8m-seg.pt").to('cuda')
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
        self.prev_dist = None
        self.prev_dist_time = None
        self.dist = None
        self.dist_time = None
        self.human_detected = False
        self.yolo_pub = rospy.Publisher('/yolo',sensor_msgs_Image)
        self.human_speed = 0
        self.rgb_img = None
        self.depth_img = None
        self.new_frame = False
        self.color = None
    
    def get_human_speed(self) -> float:
        return self.human_speed
    
    def update_human_speed(self) -> None:
        if self.prev_dist_time is None or self.dist_time is None or \
            self.prev_dist_time == self.dist_time: 
            self.human_speed = 0
        else:
            delta_dist = abs(self.dist - self.prev_dist) # meters
            delta_time = self.dist_time - self.prev_dist_time # secs
            assert delta_time > 0, 'Delta Time is negative!'
            self.human_speed = delta_dist/delta_time

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
        self.new_frame = True
        self.rgb_img = rgb_img
        self.depth_img = depth_img

    def get_newest_color_frame(self, color:sensor_msgs_Image):
        self.color = color

    def detect_human_no_depth(self):
        if self.color is None:
            return
        rgb_img = self.color
        cv_image = self.cv2_bridge.imgmsg_to_cv2(rgb_img, desired_encoding=rgb_img.encoding)
        results = self.yolo_model.predict(source=cv_image,save=True)
        # yolo_img = cv2.imread(f'{results[0].save_dir}/image0.jpg')
        # cv2.imshow('x',yolo_img)
        # cv2.waitKey(1)
        boxes = results[0].boxes # to check: results should be just 1 element when 1 img is passed to model
        self.human_detected = False
        for i, box in enumerate(boxes):
            if box.cls[0] == 0: # if box bounds a person
                self.human_detected = True
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                box_center = (b[0] + int((b[2]-b[0])/2), self.def_height-b[1] + int((b[3]-self.def_height-b[1])/2))
                self.angle = math.radians(((box_center[0] - self.img_center[0])/self.def_width)*self.FOV)

        if not self.human_detected:
            self.angle = 0

        

    def detect_human(self) -> None:
        self.new_frame = False
        if self.rgb_img is None or self.depth_img is None:
            return
        depth_img = self.depth_img
        rgb_img = self.rgb_img
        depth_cv_image = self.cv2_bridge.imgmsg_to_cv2(depth_img, desired_encoding=depth_img.encoding)
        cv_image = self.cv2_bridge.imgmsg_to_cv2(rgb_img, desired_encoding=rgb_img.encoding)
        results = self.yolo_model.predict(source=cv_image,save=True)
        # yolo_img = cv2.imread(f'{results[0].save_dir}/image0.jpg')
        # cv2.imshow('x',yolo_img)
        # cv2.waitKey(1)
        boxes = results[0].boxes # to check: results should be just 1 element when 1 img is passed to model
        human_detected = False
        for i, box in enumerate(boxes):
            if box.cls[0] == 0: # if box bounds a person
                human_detected = True
                self.mask = np.array(results[0].masks[i].data[0].cpu().detach().numpy(),dtype='bool')
                depth_mode = stats.mode(depth_cv_image[self.mask][np.nonzero(depth_cv_image[self.mask])]).mode[0]
                # if not self.dist or abs(depth_mode-self.dist) < self.dist_thresh:
                if self.prev_dist is None:
                    self.prev_dist = depth_mode
                    self.prev_dist_time = depth_img.header.stamp.secs
                else:
                    self.prev_dist = self.dist
                    self.prev_dist_time = self.dist_time
                
                self.dist = depth_mode
                self.dist_time = depth_img.header.stamp.secs
                self.update_human_speed()
                # depth_mean = np.mean(np.nonzero(depth_cv_image[self.mask]))
                # print("DEPTH MEAN= {:3f}".format(depth_mean))
                # print("DEPTH MODE= {:3f}".format(depth_mode))
                print(f'Person detected with {box.conf.item()} confidence at {self.angle} radians, {depth_mode} mm away')
                break
        if not human_detected:
            self.human_speed = 0
            self.prev_dist = None
            self.prev_dist_time = None
            self.dist = None
            self.dist_time = None
 
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
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_img_sub,self.depth_img_sub],1,0.5)
        self.ts.registerCallback(self.img_processor.get_newest_frame)
        self.rgb_img_sub_2 = rospy.Subscriber("/camera/color/image_raw", sensor_msgs_Image, self.img_processor.get_newest_color_frame)
        # self.rgb_depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw, sensor_msgs_Image, self.img_processor.get_newest_depth_frame)
        self.rate = rate

        # rospy.wait_for_message("/camera/color/image_raw", sensor_msgs_Image)

    def process(self):
        self.img_processor.detect_human()

    def check_for_human(self):
        self.img_processor.detect_human_no_depth()

    def get_angle(self):
        return self.img_processor.get_angle()
    
    def get_dist(self):
        return self.img_processor.get_dist()
    
    def get_human_speed(self):
        return self.img_processor.get_human_speed()

class Triton:
    def __init__(self, camera:Camera, rate) -> None:
        self.camera = camera
        self.rate = rate
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.vel_cmd = Twist()
        self.lidar = lp.Lidar()

        # METRICS
        self.angle_metric = []
        self.human_visible_metric = []
        self.dist_metric = []

    def move_to_human(self, goal_dist=1000, max_speed=0.5, min_speed=-0.3) -> None:
        angle = self.camera.get_angle()
        if abs(angle) > 0.1:
            self.vel_cmd.angular.z = -0.5 * angle
        else:
            self.vel_cmd.angular.z = 0
        
        behind = np.array(self.lidar.scan)[250:290]

        if not self.camera.get_dist():
            self.vel_cmd.linear.x = 0
            self.vel_pub.publish(self.vel_cmd)
            return
        dist = self.camera.get_dist() - goal_dist
        self.vel_cmd.linear.x = max(min(0.5 * (dist/1000),max_speed),min_speed)
        if np.nanmin(behind) < 0.3 and self.vel_cmd.linear.x < 0:
            self.vel_cmd.linear.x = 0
        self.vel_pub.publish(self.vel_cmd)

    # def move_to_human_var(self, goal_dist=1000, max_speed=1.3, min_speed=-0.3) -> None:
    #     angle = self.camera.get_angle()
    #     human_speed = self.camera.get_human_speed()
    #     dist = self.camera.get_dist() - goal_dist
    #     self.vel_cmd.angular.z = -0.5 * angle
    #     if human_speed > 0:
    #         dist = self.camera.get_dist()
    #         self.vel_cmd.linear.x = max(min(0.5 * (dist/1000),max_speed),min_speed)
    #     self.vel_pub.publish(self.vel_cmd)

    def stop(self):
        self.vel_cmd = Twist()
        self.vel_pub.publish(self.vel_cmd)

    def run(self) -> None:
        while not self.lidar.is_lidar_available() or self.camera.img_processor.color:
            time.sleep(1)
        times_human_not_found = 0
        while not rospy.is_shutdown():

            self.camera.check_for_human()
            if self.camera.img_processor.human_detected:

                # RECORD ANGLE
                self.human_visible_metric.append(1)
                self.angle_metric.append(self.camera.get_angle())

                times_human_not_found = 0
                # if self.camera.img_processor.new_frame:
                self.camera.process()
                self.dist_metric.append(self.camera.get_dist())
                # else:
                #     self.camera.img_processor.dist = None
                self.move_to_human()

            else:
                # RECORD LAST IN PLACE OF UNK ANGLE
                self.angle_metric.append(self.angle_metric[-1])
                self.human_visible_metric.append(0)
                self.dist_metric.append(self.dist_metric[-1])

                times_human_not_found += 1
                self.stop()
                if times_human_not_found > 5:
                    curr_robot_theta=0
                    theta_to_human = self.lidar.find_human()

                    if len(theta_to_human) > 0:
                        for theta in theta_to_human:
                            theta = math.radians(theta)

                            human_found = False
                            while abs(curr_robot_theta - theta) > 0.1:
                                self.vel_cmd.angular.z = 0.9 * (theta - curr_robot_theta)
                                self.vel_pub.publish(self.vel_cmd)
                                curr_robot_theta += 0.1 * 0.9 * (theta - curr_robot_theta)
                                self.camera.check_for_human()
                                if self.camera.img_processor.human_detected:
                                    human_found = True
                                    break
                                self.rate.sleep()
                            
                            self.stop()
                            rospy.wait_for_message('/camera/color/image_raw', sensor_msgs_Image)
                            self.camera.check_for_human()

                            if self.camera.img_processor.human_detected or human_found:
                                break

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
        pickle.dump(triton.angle_metric,open('angle_metric','wb'))
        print(e)