#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image as sensor_msgs_Image
from cv_bridge import CvBridge
import cv2
        
rospy.init_node("yolo_pub")

cv2_bridge = CvBridge()
yolo_pub = rospy.Publisher('/yolo',sensor_msgs_Image)

rate = rospy.Rate(10)

while not rospy.is_shutdown():
    try:
        yolo_img = cv2.imread(f'data/yolo/image0.jpg')
        yolo_img = cv2.resize(yolo_img, (320,240))
        yolo_img = cv2_bridge.cv2_to_imgmsg(yolo_img)
        yolo_pub.publish(yolo_img)
    except:
        pass
    rate.sleep()