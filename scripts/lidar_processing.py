#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from people_msgs.msg import PositionMeasurementArray
import math

class Lidar:
    def __init__(self) -> None:
        # rospy.init_node("lidar_processing")
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.human_pos_sub = rospy.Subscriber("/leg_tracker_measurements", PositionMeasurementArray, self.humanpos_callback)
        self.human_pos_ests = None
        self.scan = None
        # self.human_co_occur_mat = None

    def is_lidar_available(self) -> bool:
        if not self.human_pos_ests:
            return False
        if len(self.human_pos_ests) > 0:
            return True
        return False

    def scan_callback(self, data) -> None:
        self.scan = data.ranges

    def humanpos_callback(self, data) -> None:
        self.human_pos_ests = data.people
        # self.human_co_occur_mat = data.cooccurrence

    def find_human(self) -> float:
        angles = []
        reliability = []
        if len(self.human_pos_ests) > 0:
            for pos_est in self.human_pos_ests:
                pos = pos_est.pos
                x = pos.x
                y = pos.y
                reliability.append(pos_est.reliability)
                angles.append(math.degrees(math.atan2(y, x)))
            sort_idx = np.argsort(np.abs(angles))
            angles = np.array(angles)
            angles = angles[sort_idx]
        return angles