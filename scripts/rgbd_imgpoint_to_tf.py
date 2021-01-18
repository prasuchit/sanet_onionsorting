#!/usr/bin/env python
import os
# import tf
import sys
import cv2
import time
import rospy
import random
import pprint
import image_geometry
import message_filters
import numpy as np
from itertools import chain
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
# from tf import TransformListener, transformations
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from sawyer_irl_project.msg import OBlobs
from sanet_onionsorting.srv import yolo_srv


class Camera():
    def __init__(self, camera_name, rgb_topic, depth_topic, camera_info_topic, response = None):
        self.camera_name = camera_name
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.xs = response.centx
        self.ys = response.centy
        self.colors = response.color

        self.poses = []
        self.rays = []
        self.OBlobs_x = []
        self.OBlobs_y = []
        self.OBlobs_z = []

        self.pose3D_pub = rospy.Publisher('object_location', OBlobs, queue_size=1)

        # self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        # cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
        # cv2.setMouseCallback("Image window", self.mouse_callback)
        # self.br = tf.TransformBroadcaster()
        # self.lis = tf.TransformListener()
        # # Have we processed the camera_info and image yet?
        # self.ready_ = False

        tfBuffer = tf2_ros.Buffer()
        self.br = tf2_ros.TransformBroadcaster()
        self.lis = tf2_ros.TransformListener(tfBuffer)

        self.bridge = CvBridge()

        self.camera_model = image_geometry.PinholeCameraModel()

        rospy.loginfo('Camera {} initialised, {}, {}, {}'.format(self.camera_name, rgb_topic, depth_topic, camera_info_topic))

        q = 25

        self.sub_rgb = message_filters.Subscriber(rgb_topic, Image, queue_size = q)
        self.sub_depth = message_filters.Subscriber(depth_topic, Image, queue_size = q)
        self.sub_camera_info = message_filters.Subscriber(camera_info_topic, CameraInfo, queue_size = q)
        # self.tss = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth, self.sub_camera_info], queue_size=15, slop=0.4)
        self.tss = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth, self.sub_camera_info], queue_size = 30, slop = 0.5)
        #self.tss = message_filters.TimeSynchronizer([sub_rgb], 10)
        self.tss.registerCallback(self.callback)

    def callback(self, rgb_msg, depth_msg, camera_info_msg):
       
        self.camera_model.fromCameraInfo(camera_info_msg)
        # img =  np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1).astype('float32')
        # img = img/255
        img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_32FC1 = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        self.latest_depth_32FC1 = depth_32FC1.copy()

        # print("Camera model tf: ", self.camera_model.tfFrame())
        # res = kinect_utils.filter_depth_noise(depth_32FC1)
        # depth_display = kinect_utils.normalize_depth_to_uint8(depth_32FC1.copy())
        # 
        # depth_32FC1[depth_32FC1 < 0.1] = np.finfo(np.float32).max

        # cv2.imshow("Image window", img)
        # cv2.imshow("depth", depth_display)

        # cv2.setMouseCallback("Image window", self.mouse_callback)

        # cv2.waitKey(1)

        self.convertto3D()

        if len(self.poses) > 0:
            # print "Sleeping now"
            # time.sleep(200)
            self.getCam2Worldtf()
            # self.br.sendTransform(self.pose,(0,0,0,1),rospy.Time.now(),"clicked_object",self.camera_model.tfFrame())
            # self.marker_pub.publish(self.generate_marker(rospy.Time(0), self.get_tf_frame(), self.pose))
            ob = OBlobs()
            ob.x = self.OBlobs_x
            ob.y = self.OBlobs_y
            ob.z = self.OBlobs_z
            ob.color = self.colors
            self.pose3D_pub.publish(ob)
            self.poses = []
            self.rays = []  
            self.OBlobs_x = []
            self.OBlobs_y = []
            self.OBlobs_z = []

    def getCam2Worldtf(self):
        print("Camera frame is: ",self.get_tf_frame())
        for i in range(len(self.poses)):
            camerapoint =  tf2_geometry_msgs.tf2_geometry_msgs.PoseStamped()
            camerapoint.header.frame_id = self.get_tf_frame()
            camerapoint.header.stamp = rospy.Time(0)
            camerapoint.pose.position.x = self.poses[i][0]   
            camerapoint.pose.position.y = self.poses[i][1]   
            camerapoint.pose.position.z = self.poses[i][2]
            ########################################################################################
            tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) # tf buffer length
            tf_listener = tf2_ros.TransformListener(tf_buffer)
            cam_to_root_tf = tf_buffer.lookup_transform("root",
                                        self.get_tf_frame(), #source frame
                                        rospy.Time(0), # get the tf at first available time
                                        rospy.Duration(1.0)) #wait for 1 second
            tf_point = tf2_geometry_msgs.do_transform_pose(camerapoint, cam_to_root_tf)
            self.OBlobs_x.append(tf_point.pose.position.x)
            self.OBlobs_y.append(tf_point.pose.position.y)
            self.OBlobs_z.append(tf_point.pose.position.z)
	# print '2D to Camera point: \n',camerapoint    
    # print 'cam_to_root_tf: \n',cam_to_root_tf    
	# print 'Transform: \n',self.tf_point
        #########################################################################################

    def get_current_raw_image(self):
        return self.bridge.imgmsg_to_cv2(self.latest_img_msg, "bgr8")

    def get_current_rect_image(self):
        output_img = np.ndarray(self.get_current_raw_image().shape)
        self.camera_model.rectifyImage(self.get_current_raw_image(), output_img)
        return output_img

    def get_tf_frame(self):
        return self.camera_model.tfFrame()

    def is_ready(self):
        return self.ready_

    def get_ray(self, uv_rect):
        return self.camera_model.projectPixelTo3dRay(self.camera_model.rectifyPoint(uv_rect))

    def get_position_from_ray(self, ray, depth):
        """
        @brief      The 3D position of the object (in the camera frame) from a camera ray and depth value

        @param      ray    The ray (unit vector) from the camera centre point to the object point
        @param      depth  The norm (crow-flies) distance of the object from the camera

        @return     The 3D position of the object in the camera coordinate frame
        """

        # [ray_x * depth / ray_z, ray_y * depth / ray_z, ray_z * depth / ray_z]
        return [(i * depth) / ray[2] for i in ray]

    def generate_marker(self, stamp, frame_id, pose_3D):
            #marker_msg = Marker()
            #marker_msg.header.stamp = stamp
            #marker_msg.header.frame_id = frame_id
            #marker_msg.id = 0 #Marker unique ID

            ##ARROW:0, CUBE:1, SPHERE:2, CYLINDER:3, LINE_STRIP:4, LINE_LIST:5, CUBE_LIST:6, SPHERE_LIST:7, POINTS:8, TEXT_VIEW_FACING:9, MESH_RESOURCE:10, TRIANGLE_LIST:11
            #marker_msg.type = 2
            #marker_msg.lifetime = 1
            #marker_msg.pose.position = pose_3D
            marker_msg = Marker()
            marker_msg.header.frame_id = frame_id
            marker_msg.type = marker_msg.SPHERE
            marker_msg.action = marker_msg.ADD
            marker_msg.scale.x = 0.2
            marker_msg.scale.y = 0.2
            marker_msg.scale.z = 0.2
            marker_msg.color.a = 1.0
            marker_msg.color.r = 1.0
            marker_msg.color.g = 1.0
            marker_msg.color.b = 0.0
            marker_msg.pose.orientation.w = 1.0
            magicval_1 = 1.7
            marker_msg.pose.position.x = pose_3D[0]
            marker_msg.pose.position.y = pose_3D[1]
            marker_msg.pose.position.z = pose_3D[2]
            marker_msg.id = 1

            return marker_msg

    def process_ray(self, uv_rect, depth):
        ray = self.get_ray(uv_rect)
        pose = self.get_position_from_ray(ray,depth)

        # print('Ray {}\n'.format(ray))
        # print('Pose {}\n'.format(pose))
        return ray, pose

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:

            # clamp a number to be within a specified range
            clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

            #Small ROI around clicked point grows larger if no depth value found
            for bbox_width in range(20, int(self.latest_depth_32FC1.shape[0]/3), 5):
                tl_x = clamp(x-bbox_width/2, 0, self.latest_depth_32FC1.shape[0])
                br_x = clamp(x+bbox_width/2, 0, self.latest_depth_32FC1.shape[0])
                tl_y = clamp(y-bbox_width/2, 0, self.latest_depth_32FC1.shape[1])
                br_y = clamp(y+bbox_width/2, 0, self.latest_depth_32FC1.shape[1])
                # print('\n x, y, tl_x, tl_y, br_x, br_y: ',(x, y), (tl_x, tl_y, br_x, br_y))
                roi = self.latest_depth_32FC1[tl_y:br_y, tl_x:br_x]
                depth_distance = np.median(roi)

                if not np.isnan(depth_distance):
                    break

            # print('distance (crowflies) from camera to point: {:.2f}m'.format(depth_distance))
            self.ray, self.pose = self.process_ray((x, y), depth_distance)

    def convertto3D(self):

        # clamp a number to be within a specified range
        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        depth_distances = []
        # print self.xs[0], self.ys[0], self.colors[0]
        # time.sleep(300)
        #Small ROI around clicked point grows larger if no depth value found
        for i in range(len(self.xs)):
            for bbox_width in range(20, int(self.latest_depth_32FC1.shape[0]/3), 5):
                tl_x = int(clamp(self.xs[i]-bbox_width/2, 0, self.latest_depth_32FC1.shape[0]))
                br_x = int(clamp(self.xs[i]+bbox_width/2, 0, self.latest_depth_32FC1.shape[0]))
                tl_y = int(clamp(self.ys[i]-bbox_width/2, 0, self.latest_depth_32FC1.shape[1]))
                br_y = int(clamp(self.ys[i]+bbox_width/2, 0, self.latest_depth_32FC1.shape[1]))
                # print('\n x, y, tl_x, tl_y, br_x, br_y: ',(self.xs[i], self.ys[i]), (tl_x, tl_y, br_x, br_y))
                # time.sleep(200)
                roi = self.latest_depth_32FC1[tl_y:br_y, tl_x:br_x]
                depth_distances.append(np.median(roi))

                if not np.isnan(depth_distances).any():
                    break

        # print('distance (crowflies) from camera to point: {:.2f}m'.format(depth_distance))
        for i in range(len(self.xs)):
            ray, pose = self.process_ray((self.xs[i], self.ys[i]), depth_distances[i])
            self.rays.append(ray); self.poses.append(pose)
	# print '(x,y): ',self.x,self.y
	# print '3D pose: ', self.pose

def main():
    try:
        rospy.init_node('depth_from_object', anonymous=True)
        rate = rospy.Rate(10)
        rospy.wait_for_service("/get_predictions")
        # response = None
        while not rospy.is_shutdown():
            print '\nUpdating YOLO predictions...\n'
            gip_service = rospy.ServiceProxy("/get_predictions", yolo_srv)
            response = gip_service()
            if len(response.centx) > 0:
                print '\nCentroid of onions: [x1,x2...],[y1,y2...] \n',response.centx, response.centy
                #camera = Camera('usb_cam', '/kinect2/qhd/image_color', '/kinect2/qhd/camera_info')
                #NEW
                camera = Camera('kinectv2', '/kinect_V2/rgb/image_raw', '/kinect_V2/depth/image_raw', '/kinect_V2/rgb/camera_info', response)
                # camera = Camera('kinectv2', '/kinect2/hd/image_color_rect', '/kinect2/hd/image_depth_rect', '/kinect2/hd/camera_info')    # Real kinect - this works
                rospy.sleep(5)
            else:
                print '\nWaiting for detections from yolo'
            # rospy.spin()

    except rospy.ROSInterruptException:
        print("Shutting down")
        # cv2.destroyAllWindows()


if __name__ == '__main__':    
    main()