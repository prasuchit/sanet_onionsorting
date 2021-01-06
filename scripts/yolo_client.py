#!/usr/bin/env python

import rospy
from sanet_onionsorting.srv import yolo_srv
# from rgbd_imgpoint_to_tf import Camera


if __name__ == "__main__":
    rospy.init_node("yolo_client")
    rospy.wait_for_service("/get_predictions")
    rate=rospy.Rate(10)

    try:
        gip_service = rospy.ServiceProxy("/get_predictions", yolo_srv)
        while not rospy.is_shutdown():
            response = gip_service()
            print("Centroid of leftmost onion: ", response.centx, response.centy)
            # x1 = str(response.x1)
            # y1 = str(response.y1)
            # o1 = str(response.o1)
            # a1 = str(response.a1)
            # x2 = str(response.x2)
            # y2 = str(response.y2)
            # o2 = str(response.o2)
            # a2 = str(response.a2)

            # if (x1 == 'None' or x1 == '2' or y1 == '17'):
            #     pink = str('pink,(None),None')
            # else:
            #     pink = str('pink,(')+x1+','+y1+','+o1+'),'+a1

            # if (x2 == 'None' or x2 == '2' or y2 == '17'):
            #     yellow = str('yellow,(None),None')
            # else:
            #     yellow = str('yellow,(')+x2+','+y2+','+o2+'),'+a2

            # rospy.loginfo(pink)
            # rospy.loginfo(yellow)
            # rospy.loginfo("------------------")      


#rospy.loginfo('Red,(',str((response.x2,response.y2,response.o2)),'),',str(response.a2))
    except rospy.ServiceException as e:
        rospy.logwarn("Service YOLO Failed"+str(e))