"""
Inegrated code for canine orin

specifications 
    1) subscribe ros2 image topic
    2) execute object detection
    3) return position

additional requirements
    1) get position
    2) make arm movement
        i) calculate data (main function: reference arm position & reference arm euler angle)
        end effector position & euler angle (with shared memory control)
        ii) make it as pose ros topic, publish
"""

# imports: cvbridge, models, rclpy

import struct
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plot
import PIL.Image as pil_image

import cv2
from cv_bridge import CvBridge

import nanoowl.owl_predictor.OwlPredictor as owlpredictor
import nanosam.utils.predictor.Predictor as sampredictor

import rclpy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from rclpy.qos import QoSProfile



class POSE_RETURNER(node):
    
    def __init__(self, args):

        """ROS settings - subscriber and publisher"""

        super().__init__("pose_returner")
        qos_profile = QoSProfile(depth=10)

        # raw image subscriber
        self.subscriber_raw = self.create_subscription(
            Image,
            "/zed/zed_node/rgb_raw/image_raw_color",
            self.image_callback
            )
        self.subscriber_raw

        # point cloud subscriber
        self.subscriber_cloud = self.create_subscription(
            PointCloud2,
            "/zed/zed_node/point_cloud/cloud_registered", #zstd, zlib, draco
            self.cloud_callback
            )
        self.subscriber_cloud
        
        # cv bridge object
        self.bridge = CvBridge()

        # pose publisher
        self.publisher = self.create_publisher(
            Pose,
            "/user/obj_pose",
            qos_profile
        )

        # timer for publish
        self.timer = self.create_timer(10, self.pub_pose)



        """arguments parsing & storing"""
        # threshold parse
        self.thresholds = args.threshold.strip("][()")
        self.thresholds = self.thresholds.split(',')
        if len(self.thresholds) == 1:
            self.thresholds = float(self.thresholds[0])
        else:
            self.thresholds = [float(x) for x in self.thresholds]
        
        # prompt parsing: text to list of texts
        self.prompt = args.prompt.strip("][()")
        self.prompt = self.prompt.split(',') # done


        """define models: OWL ViT, SAM"""
        self.detector = owlpredictor(
            args.model,
            image_encoder_engine=args.image_encoder_engine
        )
        
        self.sam_model = sampredictor(
            args.image_encoder,
            args.mask_decoder
        ) # in sam example code, it uses parser arguments but this is also possible -> not available..




    def image_callback(self, msg):
        # owl forwarding, check validity (if object exists)

        # use cv bridge to convert
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image_feed = pil_image.fromarray(frame).convert("RGB")

        output = self.detector.predict(
                image=image_feed, 
                text = self.prompt, 
                text_encodings=text_encodings,
                threshold=self.thresholds,
                pad_square=False
            )

        # check validity (check if only one object detected)
        N = len(output.labels)

        if N == 0:
            print("no object detected") # invalid
            self.bbox = None
            self.position_2d = None

        elif N > 1:
            print("multiple objects detected") # invalid
            self.bbox = None
            self.position_2d = None

        else:
            owl_box = output.boxes[0]
            self.bbox = np.array([int(x) for x in owl_box]) # bounding box numpy, integer

            # SAM forwarding, according to validity

            # image set, forwarding
            self.sam_model.set_image(image_feed)
            points, point_labels = bbox2points(self.bbox)

            # get mask & refined mask
            mask, _, _ = self.sam_model.predict(points, point_labels)
            mask = (mask[0, 0] > 0).detach().cpu().numpy() # boolean mask

            # return 2d position from mask (refined: detached & float to boolean)
            mask = mask.astype("uint8")

            # create mesh, calculate center position of mask
            mesh_y, mesh_x = np.mgrid[0:mask.shape[0]:1, 0:mask.shape[1]:1]

            # bitwise AND between mesh and mask
            if np.count_nonzero(mask):
                centroid_x = int(float(np.sum(cv2.bitwise_and(mesh_x, mesh_x, mask=mask))/np.count_nonzero(mask)))
                centroid_y = int(float(np.sum(cv2.bitwise_and(mesh_y, mesh_y, mask=mask)))/np.count_nonzero(mask))
                #by using bitwise AND, only true-masked pixels alive -> sum them all!
            
            else: #mask is not valid
                centroid_x = mask.shape[1] /2
                centroid_y = mask.shape[0] /2

            # save 2d centroid
            # coordinate: not camera coordinate, mesh grid coordinate (matrix-wise)
            self.position_2d = np.array([centroid_x, centroid_y])



    def cloud_callback(self, msg): # calculate 3d position using ROS2 pointcloud topic

        # check 2d position validity
        if self.position_2d is not None:

            if msg.height != 1: # shape received (2d data)
                pass
            else: # insert proper value
                msg.height = 1080
                msg.width = 1920
            
            # calculate 3d position with pcl
            point = pc2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
                uvs=[self.position_2d[0],self.position_2d[1]]
                )
            # returns yielded points (uv given, 1 value returned)

            print(point) # check

            self.position_3d = np.array(point)


        else: # if 2d position not valid (None) -> 3d position None
            self.get_logger().info("position invalid")
            self.position_3d = None

        
    def pub_pose(self):

        # define ros2 topic to publish
        pose = Pose()

        if self.position_3d is not None: # if position exists, publish pose
            pose.position.x = self.position_3d[0]
            pose.position.y = self.position_3d[1]
            pose.position.z = self.position_3d[2]

        else: # if not valid -> inf?
            inf_num = float('inf')
            pose.position.x = inf_num
            pose.position.y = inf_num
            pose.position.z = inf_num
        
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 30 # correct value

        self.publisher.publish(pose)



# bounding box function
def bbox2points(bbox):
    points = np.array([
        [bbox[0], bbox[1]],
        [bbox[2], bbox[3]]
    ])
    point_labels = np.array([2, 3])
    return points, point_labels


def main():

    # arguments declaration
    parser = argparse.ArgumentParser()
    # 1-1. sam arguments
    parser.add_argument("--image_encoder", type=str, default="engines/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="engines/mobile_sam_mask_decoder.engine")
    # 1-2. owl arguments
    parser.add_argument("--threshold", type=str, default="0.1,0.1")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="engines/owl_image_encoder_patch32.engine")
    parser.add_argument("--prompt", type=str, default="a can")
    # change to arguments
    arguments = parser.parse_args()

    # rclpy part
    rclpy.init(args=None)
    node = POSE_RETURNER(arguments)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
