#! /usr/bin/python

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
import sys
sys.path.append('/home/psuresh/catkin_ws/src/sanet_onionsorting/')
from networkconfig import frontend_config as fc
from thirdparty.RCN.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from backend import nnn

# minor imports
import json
import os
import numpy as np
import cv2
import math


class getImgPredictions:
    model = keras.Sequential()
    initframe = 0

    # Memory related variables
    MEMORY_FLAG = False
    memoryframet1_rgb = None
    memoryframet2_rgb = None
    memoryframet1_depth = None
    memoryframet2_depth = None
    xymemory = None

    # rcnch -> Object detection strategy
    def __init__(self, rcnch='yolo'):
        self.rcnch = rcnch
        self.network = nnn()
        configs = fc.configs_raw()
        # self.maskmemory = np.load(configs.mask_rgb_0)
        # self.maskmemory_depth = np.load(configs.mask_depth_0)

        if self.rcnch == 'faster':
            from thirdparty.RCN.keras_retinanet import models as rcn_model
            print("Faster RCN - [ Initilizing ]")
            self.rcn = rcn_model.load_model(configs.snap_path, backbone_name='resnet50')
            print("Faster RCN - [ Done ]")

        # keras.backend.tensorflow_backend.set_session(tf.Session())
        if self.rcnch == 'yolo':
            print("YOLO - [ Initilizing ]")
            from thirdparty.yolo.frontend import YOLO
            
            with open(configs.config_path) as config_buffer:
                config = json.load(config_buffer)
            self.yolo = None
            self.yolo = YOLO(backend=config['model']['backend'],
                             input_size=(config['model']['input_size_h'], config['model']['input_size_w']),
                             labels=config['model']['labels'],
                             max_box_per_image=config['model']['max_box_per_image'],
                             anchors=config['model']['anchors'],
                             gray_mode=config['model']['gray_mode'])
            print("YOLO - [ Loading Weights ]")
            self.yolo.load_weights(configs.yolo_weight_file)
            print("YOLO - [ Complete ]")


    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    # def remove_redundant(self, robot1box, robot2box, robot1_score, robot2_score):

    #     if self.bb_intersection_over_union(robot1box, robot2box) > 0.2:
    #         if robot1_score > robot2_score:
    #             return 0
    #         else:
    #             return 1
    #     else:
    #         return 3

    # # RCNN is taken from retinanet in github
    # def getObjectLocation_multi(self, image):
    #     '''

    #     :param image:
    #     :return: robot1, robot2, image
    #     '''
    #     imagep = preprocess_image(image)
    #     image, scale = resize_image(imagep)
    #     labels_to_names = {0: 'turtlebot_Pink', 1: 'turtlebot_Yellow'} # have to redo
    #     robot1 = None
    #     robot2 = None
    #     score_rb1 = 0
    #     score_rb2 = 0
    #     boxes, scores, labels = self.rcn.predict_on_batch(np.expand_dims(image, axis=0))
    #     boxes /= scale
    #     done_flag_rb1 = 0
    #     done_flag_rb2 = 0
    #     for box, score, label in zip(boxes[0], scores[0], labels[0]):
    #         # scores are sorted so we can break
    #         if score < 0.1:
    #             break
    #         if done_flag_rb2 + done_flag_rb1 >= 2:
    #             break
    #         # A list of 4 elements (x1, y1, x2, y2)
    #         b = np.array(box).astype(int)
    #         if labels_to_names[label] == "turtlebot_Pink" and done_flag_rb2 == 0:
    #             robot1 = b
    #             score_rb2 = score
    #             #print("robot2:" + str(score))
    #             done_flag_rb2 = 1
    #         elif labels_to_names[label] == "turtlebot_Yellow" and done_flag_rb1 == 0:
    #             robot2 = b
    #             #print("robot1:" + str(score))
    #             score_rb1 = score
    #             done_flag_rb1 += 1
    #     if robot2 is not None and robot1 is not None:
    #         redudant = self.remove_redundant(robot2, robot1, score_rb1, score_pink)
    #         if redudant == 0:
    #             robot1 = None
    #         elif redudant == 1:
    #             robot2 = None
    #     print ('========== getObjectLocation_multi ==========')
    #     print ('pink ::: ', robot1)
    #     print ('yellow ::: ', robot2)
    #     return robot2, robot1, imagep

    def getObjectLocation_yolo(self, image):
        # adjust this to point to your downloaded/trained model
        print("Detetcting Scene Objects using YOLO")
        image_h, image_w, _ = image.shape
        boxes = self.yolo.predict(image)
        labels_to_names = {0: 'Blemished', 1: 'Unblemished'}
        if len(boxes) <= 0:
            return (None, image) 

        # scores_r1 = []
        # for i in range(len(boxes)):
        #     if(str(boxes[i].get_label()) == '0'):
        #         scores_r1.append(boxes[i].get_score())
        # if len(scores_r1) > 0:
        #     c = np.argmax(scores_r1)
        #     xmin_r1 = int(boxes[c].xmin * image_w)
        #     ymin_r1 = int(boxes[c].ymin * image_h)
        #     xmax_r1 = int(boxes[c].xmax * image_w)
        #     ymax_r1 = int(boxes[c].ymax * image_h)
        # else:
        #     xmin_r1 = -1
        #     ymin_r1 = -1
        #     xmax_r1 = -1
        #     ymax_r1 = -1

        # scores_r2 = []
        # for i in range(len(boxes)):
        #     if(str(boxes[i].get_label()) == "1"):
        #         scores_r2.append(boxes[i].get_score())
        # if len(scores_r2) > 0:
        #     c = np.argmax(scores_r2)
        #     xmin_r2 = int(boxes[c].xmin * image_w)
        #     ymin_r2 = int(boxes[c].ymin * image_h)
        #     xmax_r2 = int(boxes[c].xmax * image_w)
        #     ymax_r2 = int(boxes[c].ymax * image_h)
        # else:
        #     xmin_r2 = -1
        #     ymin_r2 = -1
        #     xmax_r2 = -1
        #     ymax_r2 = -1

        # print("------------")
        # print ('========== getObjectLocationLocation_yolo ==========')
        # print ('pink ::: ', np.array([xmin_r1, ymin_r1, xmax_r1, ymax_r1]))
        # print ('yellow ::: ', np.array([xmin_r2, ymin_r2, xmax_r2, ymax_r2]))
        # return np.array([xmin_r1, ymin_r1, xmax_r1, ymax_r1]), np.array([xmin_r2, ymin_r2, xmax_r2, ymax_r2]),image
        return boxes, image

    # def manualtester(self):
    #     while True:
    #         n = int(input("Enter index:"))
    #         print(self.network.actionlables[n])
    #         print(self.network.statelables[n])
    #         _, _, _, _, _, _ = self.getsa(self.network.dataset[n][0])
    #         _, _, _, _, _, _ = self.getsa(self.network.dataset[n][1])
    #         predictionaction, predictionstateX, predictionstateY, predictionstateO, xy, location = self.getsa(self.network.dataset[n][2])
    #         print(n, " Action : ", predictionaction, " State : ", "( ", str(predictionstateX), ",", str(predictionstateY), ",", str(predictionstateO), ")", " XY :", xy, "Loc",
    #               str(location))

    # def getsa(self, ft_rgb, ft_d):

    #     if self.MEMORY_FLAG == True:
    #         xy = None
    #         o_image_r_t = None
    #         try:
    #             # cropImage_robot1, xy_robot1, o_image_robot1, cropImage_robot2, xy_robot2, o_image_robot2
    #             print("YOLOOOOOO   11")
    #             r_image_t, xy, o_image_r_t,_,_,_ = self.cropnmaskdata(ft_rgb, 0.1041)
    #         except Exception as ex:
    #             if "Turtlebot" in ex.args[0]:
    #                 self.maskmemory = ft_rgb
    #                 return "No TurtleBot", "No TurtleBot", "No TurtleBot", "No TurtleBot", "No TurtleBot", "No TurtleBot"
    #             else:
    #                 print(ex)
    #                 # exit(0)
    #         if min(xy) > 0:
    #             r_image_t1, o_image_r_t1 = self.crop_silentdata(self.memoryframet1_rgb, 0.1041, xy)
    #             r_image_t2, o_image_r_t2 = self.crop_silentdata(self.memoryframet2_rgb, 0.1041, xy)
    #             # Get all Depth Data for one set ch : 4

    #             d_image_t, o_image_d_t = self.crop_depthdata(ft_d, 0.1041, xy)
    #             d_image_t1, o_image_d_t1 = self.crop_depthdata(self.memoryframet1_depth, 0.1041, xy)
    #             d_image_t2, o_image_d_t2 = self.crop_depthdata(self.memoryframet2_depth, 0.1041, xy)

    #             o_image_d_t = o_image_d_t[:, :, None]
    #             o_image_d_t1 = o_image_d_t1[:, :, None]
    #             o_image_d_t2 = o_image_d_t2[:, :, None]
    #             d_image_t = d_image_t[:, :, None]

    #             predictionaction, predictionstateX, predictionstateY, predictionstateO = self.network.detect(o_image_r_t, o_image_d_t, o_image_r_t1, o_image_d_t1, o_image_r_t2,
    #                                                                                                             o_image_d_t2, r_image_t, d_image_t)

    #             self.memoryframet2_rgb = self.memoryframet1_rgb
    #             self.memoryframet2_depth = self.memoryframet1_depth

    #             self.memoryframet1_rgb = ft_rgb
    #             self.memoryframet1_depth = ft_d

    #             return predictionaction, predictionstateX, predictionstateY, predictionstateO, xy , None
    #         else:
    #             return -1, -1, -1, -1, -1 , None


    #     else:
    #         if self.initframe == 0:
    #             self.memoryframet2_rgb = ft_rgb
    #             self.memoryframet2_depth = ft_d

    #             self.initframe += 1


    #         elif self.initframe == 1:
    #             self.memoryframet1_rgb = ft_rgb
    #             self.memoryframet1_depth = ft_d

    #             self.MEMORY_FLAG = True

    #     return "stored in memory", "None", "None", "None", "None", "None"

    # def cropnmaskdata(self, image, cropfactor, maxcrop=50):
        
    #     Image_robot2 = Image_robot1 = o_image_robot1 = o_image_robot2 = None
    #     if self.rcnch == 'yolo':
    #         xy_robot1, xy_robot2, imageNum = self.getObjectLocation_yolo(image)
            
    #     else:
    #         xy_robot1, xy_robot2, imageNum = self.getObjectLocation_multi(image) #RCN

    #     if xy_robot1 is not None and np.min(xy_robot1) > 7:
    #         x = xy_robot1[3] - xy_robot1[1]
    #         crop = math.ceil((-cropfactor) * x + maxcrop)
    #         maskImage = np.copy(self.maskmemory) #rgb
    #         o_image_robot1 = image[xy_robot1[1] - 7:xy_robot1[3] + 7, xy_robot1[0] - 7:xy_robot1[2] + 7]
    #         maskImage[xy_robot1[1] - 7:xy_robot1[3] + 7, xy_robot1[0] - 7:xy_robot1[2] + 7] = o_image_robot1
    #         Image_robot1 = cv2.resize(maskImage, (640, 480), interpolation=cv2.INTER_CUBIC)
    #         o_image_robot1 = cv2.resize(o_image_robot1, (100, 150), interpolation=cv2.INTER_CUBIC)

    #     if xy_robot2 is not None and np.min(xy_robot2) > 7:
    #         x = xy_robot2[3] - xy_robot2[1]
    #         crop = math.ceil((-cropfactor) * x + maxcrop)
    #         maskImage = np.copy(self.maskmemory)
    #         o_image_robot2 = image[xy_robot2[1] - 7:xy_robot2[3] + 7, xy_robot2[0] - 7:xy_robot2[2] + 7]
    #         maskImage[xy_robot2[1] - 7:xy_robot2[3] + 7, xy_robot2[0] - 7:xy_robot2[2] + 7] = o_image_robot2
    #         Image_robot2 = cv2.resize(maskImage, (640, 480), interpolation=cv2.INTER_CUBIC)
    #         o_image_robot2 = cv2.resize(o_image_robot2, (100, 150), interpolation=cv2.INTER_CUBIC)

    #     if xy_robot2 is None and xy_robot1 is None:
    #         raise Exception("No robot1 and robot2 Turtlebot")
    #     return Image_robot1, xy_robot1, o_image_robot1, Image_robot2, xy_robot2, o_image_robot2

    # def crop_silentdata(self, image, cropfactor, xy, maxcrop=50):

    #     x = xy[3] - xy[1]
    #     crop = math.ceil((-cropfactor) * x + maxcrop)
    #     cropImage = np.copy(self.maskmemory)
    #     cropImage[xy[1] - 7:xy[3] + 7, xy[0] - 7:xy[2] + 7] = image[xy[1] - 7:xy[3] + 7, xy[0] - 7:xy[2] + 7]
    #     o_image = image[xy[1] - 7:xy[3] + 7, xy[0] - 7:xy[2] + 7]
    #     cropImage = cv2.resize(cropImage, (640, 480), interpolation=cv2.INTER_CUBIC)
    #     o_image = cv2.resize(o_image, (100, 150), interpolation=cv2.INTER_CUBIC)
    #     return cropImage, o_image

    # def crop_depthdata(self, depth, cropfactor, xy, maxcrop=50):

    #     x = xy[3] - xy[1]
    #     crop = math.ceil((-cropfactor) * x + maxcrop)
    #     cropImage = np.copy(self.maskmemory_depth)
    #     cropImage = np.squeeze(cropImage)
    #     cropImage[xy[1] - 7:xy[3] + 7, xy[0] - 7:xy[2] + 7] = depth[xy[1] - 7:xy[3] + 7, xy[0] - 7:xy[2] + 7]
    #     o_image = depth[xy[1] - 7:xy[3] + 7, xy[0] - 7:xy[2] + 7]
    #     cropImage = cv2.resize(cropImage, (640, 480), interpolation=cv2.INTER_CUBIC)
    #     o_image = cv2.resize(o_image, (100, 150), interpolation=cv2.INTER_CUBIC)
    #     return cropImage, o_image

    # def getsa_multi(self, ft_rgb, ft_d):
    #     '''

    #     :param ft_rgb: #ft_rgb -> frame time t rgb
    #     :param ft_d: ft_d -> frame time t depth
    #     :return: state and action
    #     '''
    #     labels = {0: 'turtlebot_1', 1: 'turtlebot_2'}
    #     if (ft_rgb is None) or (ft_d is None):
    #         return ["No TurtleBot", "No TurtleBot"], labels
    #     # MEMORY FLAG set at false means there is no prior frame data
    #     if self.MEMORY_FLAG == True:
    #         xy = None
    #         #o_image_r_t_robot1 and robot2 are the action data frames in RGB
    #         o_image_r_t_robot1 = o_image_r_t_robot2 = None
    #         try:
    #             # cropImage_robot1, xy_robot1, o_image_robot1, cropImage_robot2, xy_robot2, o_image_robot2
    #             r_image_t_robot1, xy_robot1, o_image_r_t_robot1, r_image_t_robot2, xy_robot2, o_image_r_t_robot2 = self.cropnmaskdata(ft_rgb, 0.1041)
    #         except Exception as ex:
    #             if "Turtlebot" in ex.args[0]:
    #                 self.maskmemory = ft_rgb
    #                 return ["No TurtleBot", "No TurtleBot"], labels
    #             else:
    #                 import traceback
    #                 print(str(ex.args[0]) + "\n" + str(repr(ex)))
    #                 traceback.print_exc()
    #                 exit(0)
    #         prediction = []
    #         # print("r1= "+ str(xy_robot1))
    #         # print("r2= "+ str(xy_robot2))
    #         if xy_robot1 is not None:
    #             if min(xy_robot1) > 7:
    #                 # robot1
    #                 # r_image_t1_robot1 rgb image for t-1 for robot1 turtlebot
    #                 # o_image_r_t1_robot1 cropped rgb image for t-1 for robot1 turtlebot
    #                 r_image_t1_robot1, o_image_r_t1_robot1 = self.crop_silentdata(self.memoryframet1_rgb, 0.1041, xy_robot1)
    #                 r_image_t2_robot1, o_image_r_t2_robot1 = self.crop_silentdata(self.memoryframet2_rgb, 0.1041, xy_robot1)
    #                 # Get all Depth Data for one set ch : 4

    #                 d_image_t_robot1, o_image_d_t_robot1 = self.crop_depthdata(ft_d, 0.1041, xy_robot1)
    #                 d_image_t1_robot1, o_image_d_t1_robot1 = self.crop_depthdata(self.memoryframet1_depth, 0.1041, xy_robot1)
    #                 d_image_t2_robot1, o_image_d_t2_robot1 = self.crop_depthdata(self.memoryframet2_depth, 0.1041, xy_robot1)

    #                 o_image_d_t_robot1 = o_image_d_t_robot1[:, :, None]
    #                 o_image_d_t1_robot1 = o_image_d_t1_robot1[:, :, None]
    #                 o_image_d_t2_robot1 = o_image_d_t2_robot1[:, :, None]
    #                 d_image_t_robot1 = d_image_t_robot1[:, :, None]
    #                 predictionaction_robot1, predictionstateX_robot1, predictionstateY_robot1, predictionstateO_robot1 = self.network.detect(o_image_r_t_robot1, o_image_d_t_robot1, o_image_r_t1_robot1,
    #                                                                                                                                 o_image_d_t1_robot1, o_image_r_t2_robot1, o_image_d_t2_robot1,
    #                                                                                                                                 r_image_t_robot1, d_image_t_robot1)
    #                 prediction.append([predictionaction_robot1, predictionstateX_robot1, predictionstateY_robot1, predictionstateO_robot1, xy_robot1,0])
    #             else:
    #                 prediction.append([-1, -1, -1, -1, -1,-1])

    #         if xy_robot2 is not None:
    #             # robot2
    #             if min(xy_robot2) > 7:
    #                 r_image_t1_robot2, o_image_r_t1_robot2 = self.crop_silentdata(self.memoryframet1_rgb, 0.1041, xy_robot2)
    #                 r_image_t2_robot2, o_image_r_t2_robot2 = self.crop_silentdata(self.memoryframet2_rgb, 0.1041, xy_robot2)
    #                 # Get all Depth Data for one set ch : 4

    #                 d_image_t_robot2, o_image_d_t_robot2 = self.crop_depthdata(ft_d, 0.1041, xy_robot2)
    #                 d_image_t1_robot2, o_image_d_t1_robot2 = self.crop_depthdata(self.memoryframet1_depth, 0.1041, xy_robot2)
    #                 d_image_t2_robot2, o_image_d_t2_robot2 = self.crop_depthdata(self.memoryframet2_depth, 0.1041, xy_robot2)

    #                 o_image_d_t_robot2 = o_image_d_t_robot2[:, :, None]
    #                 o_image_d_t1_robot2 = o_image_d_t1_robot2[:, :, None]
    #                 o_image_d_t2_robot2 = o_image_d_t2_robot2[:, :, None]
    #                 d_image_t_robot2 = d_image_t_robot2[:, :, None]

    #                 predictionaction_robot2, predictionstateX_robot2, predictionstateY_robot2, predictionstateO_robot2 = self.network.detect(o_image_r_t_robot2, o_image_d_t_robot2, o_image_r_t1_robot2,
    #                                                                                                                                 o_image_d_t1_robot2, o_image_r_t2_robot2,
    #                                                                                                                                 o_image_d_t2_robot2,
    #                                                                                                                                 r_image_t_robot2, d_image_t_robot2)
    #                 prediction.append([predictionaction_robot2, predictionstateX_robot2, predictionstateY_robot2, predictionstateO_robot2, xy_robot2,1])
    #             else:
    #                 prediction.append([-1, -1, -1, -1, -1,-1])

    #         self.memoryframet2_rgb = self.memoryframet1_rgb
    #         self.memoryframet2_depth = self.memoryframet1_depth

    #         self.memoryframet1_rgb = ft_rgb
    #         self.memoryframet1_depth = ft_d

    #         return prediction ,labels


    #     else:
    #         if self.initframe == 0: # Do thsi first
    #             self.memoryframet2_rgb = ft_rgb
    #             self.memoryframet2_depth = ft_d

    #             self.initframe += 1


    #         elif self.initframe == 1:
    #             self.memoryframet1_rgb = ft_rgb
    #             self.memoryframet1_depth = ft_d

    #             self.MEMORY_FLAG = True

    #     return [ "stored in memory","stored in memory"], None