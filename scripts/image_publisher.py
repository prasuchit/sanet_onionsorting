#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import cv2
import numpy as np

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray

class ImagePublisher():
    def __init__(self):     
        imgarrpub = rospy.Publisher('imgarr', Int32MultiArray, queue_size=2)
        # imgarrpubsd = rospy.Publisher('imgarrsd', Int32MultiArray, queue_size=2)
        cvb = CvBridge()


    def imnormalize(self, xmax, image):
        """
        Normalize a list of sample image data in the range of 0 to 1
        : image:image data.
        : return: Numpy array of normalize data
        """
        xmin = 0
        a = 0
        b = 255

        return ((np.array(image, dtype=np.float32) - xmin) * (b - a)) / (xmax - xmin)

    def __numpy_to_string(self,A):
        return ','.join(str(x) for x in A)


    def __string_to_numpy(self,S):
        return np.array([int(x) for x in S.split(',')])


    def grabrgb(self, msg):
        try:
            cv_image = self.cvb.imgmsg_to_cv2(msg, "rgb8")  # could replace "rgb8" with msg.encoding
            print("Image_publisher node received hd image from camera topic")
            arr_img = np.array(cv_image)
            print(arr_img.shape)
            flat_arr = np.ravel(arr_img)
            msg = Int32MultiArray(data=flat_arr)
            global pubarrimg
            pubarrimg.publish(msg)
        except CvBridgeError as e:
            print(e)
        return


    # def grabrgb_sd(msg):
    #     try:
    #         cv_image = cvb.imgmsg_to_cv2(msg, "rgb8")
    #         print("Image_publisher node received sd image from camera topic")
    #         arr_img = np.array(cv_image)
    #         print(arr_img.shape)
    #         flat_arr = np.ravel(arr_img)
    #         msg = Int32MultiArray(data=flat_arr)
    #         global pubarrimg_sd
    #         pubarrimg_sd.publish(msg)
    #     except CvBridgeError as e:
    #         print(e)
    #     return


def imglistener():

    ip = ImagePublisher()
    rospy.init_node('image_publisher', anonymous=True)

    rospy.Subscriber("/kinect_V2/rgb/image_raw", Image, ip.grabrgb)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':

    imglistener()
