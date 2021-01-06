#!/usr/bin/env python3
# coding: utf8
import sys

# append py2 in order to import rospy
sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image, CameraInfo
# in order to import yolov5 under python3
sys.path.remove('/usr/lib/python2.7/dist-packages')
from sanet_onionsorting.srv import yolo_srv
import numpy as np
import copy
# from frontend import getImgPredictions
sys.path.append('/home/psuresh/catkin_ws/src/sanet_onionsorting/')
from thirdparty.yolov5.detect import YOLO
# os.system("pwd")
# import cv2
# from cv_bridge import CvBridge,CvBridgeError

# cvb = CvBridge()

# mem_action = 0
# mem_x = None
# mem_y = None
# mem_theta = None
same_flag = 0
rgb_mem = None
depth_mem = None
weights = None

def grabrgb(msg):

    global rgb_mem
    if msg is not None:
        rgb_mem = copy.copy(msg)
    else:
        return
    # global video
    # global same_flag
    # try:
    #     cv_image = cvb.imgmsg_to_cv2(msg, "rgb8")

    # except CvBridgeError as e:
    #     print("CVBridge error. Exception: ",e)

    # image_normal = np.array(cv_image)
    # if np.array_equal(rgb_mem, image_normal):
    #     same_flag = 1
    #     return
    # else:
    #     same_flag = 0


# def grabdepth(msg):
#     global depth_mem
#     if msg is not None:
#         depth_mem = copy.copy(msg)
#     else:
#         return
#     global video
#     global same_flag
#     try:
#         cv_image = cvb.imgmsg_to_cv2(msg, msg.encoding)
#     except CvBridgeError as e:
#         print("CVBridge error. Exception: ",e)

#     image_normal = np.array(imnormalize(np.max(cv_image), cv_image), dtype=np.uint8)
#     numpy_image = np.array(cv_image, dtype=np.uint16)

#     if (depth_mem == numpy_image).all():
#         same_flag = 1
#         return
#     else:
#         depth_mem = np.copy(numpy_image)

# def is_number(s):
#     try:client
#     if same_flag == 0:
#         import traceback
#         try:
#             prediction, lables = gas.getsa_multi(rgb_mem, depth_mem)
#             if ( is_number(prediction[0][0])== True and is_number(prediction[1][0])== True):
#                 return str(prediction[0][0]), str(prediction[0][1]),str(prediction[0][2]),str(prediction[0][3]),str(prediction[1][0]), str(prediction[1][1]),str(prediction[1][2]),str(prediction[1][3])
#     /camera/depth_registered/image_raw        elif is_number(prediction[0][0])== True :
#                 return str(prediction[0][0]), str(prediction[0][1]),str(prediction[0][2]),str(prediction[0][3]), "None", "None", "None", "None"
#             elif is_number(prediction[1][0])== True :
#                 return 'None', "None", "None", "None", str(prediction[1][0]), str(prediction[1][1]), str(prediction[1][2]),str(prediction[1][3]) 
#             else:
#                 return "None"," import *
#         # The old values will be used
#         pass


def getpred(msg):
    global weights, rgb_mem, depth_mem
    # print("Entered getpred func")
    bound_box_xy = []
    centxs = []
    centys = []
    colors = []
    y = YOLO(weights)
    if rgb_mem is not None: 
        # thisimage = np.frombuffer(rgb_mem.data, dtype=np.uint8).reshape(rgb_mem.height, rgb_mem.width, -1).astype('float32')
        # print("\nThis image shape: \n",np.shape(thisimage))
        output = y.detect(rgb_mem)
        print('output:   ',output)
        if output is not None and len(output) > 0:   
            for det in output:
                for *xyxy, conf, cls in det:
                            ''' 
                            NOTE: Useful link: https://miro.medium.com/max/597/1*85uPFWLrdVejJkWeie7cGw.png
                            Kinect image resolution is (1920,1080)
                            But numpy image shape is (1080,1920) becasue np takes image in the order height x width.
                            '''
                            tlx, tly, brx, bry = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            centx, centy = int((tlx+brx)/2), int((tly+bry)/2)
                            if cls in [0,1]: 
                                print("\ntlx, tly, brx, bry, cls: ",tlx, tly, brx, bry, cls)
                                # print(f"\nCentroid: {centx}, {centy}")
                                centxs.append(centx)
                                centys.append(centy)
                                colors.append(cls)
        else: print("\nNo output from yolo received yet\n")
        rgb_mem = None
        return centxs,centys,colors
    else:
        print("\nNo RGB image received yet\n")
        return None, None,None


def main():
    global weights
    try:
        rospy.init_node("yolo_service")
        rospy.loginfo("Yolo service started")
        if len(sys.argv) < 2:
            weights = "best_realkinect.pt"   # Default weights
        else:
            choice = sys.argv[1]

        if (choice == "real"):
            weights = "best_realkinect.pt"
            # for kinect v2
            print(f"{weights} weights selected with real kinect")
            rospy.Subscriber("/kinect2/hd/image_color", Image, grabrgb)
            # for kinect v2
            # rospy.Subscriber("/kinect2/hd/points", Image, grabdepth)
        elif (choice == "gazebo"):
            weights = "best_gazebokinect.pt"
            # for kinect gazebo
            print(f"{weights} weights selected with gazebo kinect")
            rospy.Subscriber("/kinect_V2/rgb/image_raw", Image, grabrgb)
            # for kinect gazebo
            # rospy.Subscriber("/kinect_V2/depth/points", Image, grabdepth)
        else:
            print(f"Unknown choice: {choice}. Please choose between real and gazebo.")

        service = rospy.Service("/get_predictions", yolo_srv, getpred)
    except rospy.ROSInterruptException:
        print(rospy.ROSInterruptException)
        return
    except KeyboardInterrupt:
        return
    rospy.spin()


if __name__ == '__main__':    
    main()
