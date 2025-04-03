"""
Integration of ZED and SAM
    1) get left image & depth image
    2) forward left image to nanoSAM model
    3) get mask & depth image -> integrate

problems (possible)
    1) difference between left image and depth image
"""

import sys
import numpy as np
import pyzed.sl as sl
import cv2
import PIL
from nanosam.utils.predictor import Predictor
#import nanosam.utils.predictor.Predictor as Pd
#no module name duplicate issue
import argparse
import matplotlib.pyplot as plot
import time


""" 1. define camera & model / parser """
parser = argparse.ArgumentParser()
parser.add_argument("--image_encoder", type=str, default="engines/resnet18_image_encoder.engine")
parser.add_argument("--mask_decoder", type=str, default="engines/mobile_sam_mask_decoder.engine")
args = parser.parse_args()

sam_model = Predictor(
    args.image_encoder,
    args.mask_decoder
)




def main():
    zed_cam = sl.Camera()

    # Set configuration & runtime parameters, open camera
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        # sys.argv: script arguments
        # 0: script file name
        # after this: arguments from command line
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NEURAL# Mode: performance, neural, etc
    init.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed_cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed_cam.close()
        exit(1)
    # Set runtime parameters after opening the camera
    runtime_param = sl.RuntimeParameters()

    """ 2. image size definition (resolution 1/2), declare sl matrices"""
    image_size = zed_cam.get_camera_information().camera_configuration.resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    left_image_mat = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_mat = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

    #set bounding box
    box = [320, 180, 640, 360] #x0, y0, x1, y1
    points = np.array([[box[0], box[1]],
                        [box[2], box[3]]])
    point_labels = np.array([2, 3])

    """3. repeat -> get image, forwarding, integration"""
    while True:
        err = zed_cam.grab(runtime_param)

        if err == sl.ERROR_CODE.SUCCESS:
            # a. retrieve left and depth image!
            zed_cam.retrieve_image(left_image_mat, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed_cam.retrieve_image(depth_image_mat, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)

            # b. get image matrix from retrieved data
            left_image = left_image_mat.get_data()
            #use PIL to feed model
            left_rgb = PIL.Image.fromarray(left_image).convert("RGB")
            # print(left_image.shape)
            # left_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGRA2BGR)
            depth_image = depth_image_mat.get_data()

            # c. model forwarding: set image, predict and return mask
            sam_model.set_image(left_rgb)
            
            # here, should load bounding box points and labels 
            mask, _, _ = sam_model.predict(points, point_labels)

            # results: depth_image & mask
            # visualize -> usually with matplotlib

            mask_refined = (mask[0, 0] > 0).detach().cpu().numpy()

            #image show with opencv
            # if mask, generate image added
            # if no mask, just view raw image

            plain_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
            mask_color = (mask_refined * 255).astype(np.uint8) if mask_refined.max() <= 1 else binary_mask

            color_mask = np.zeros_like(plain_image)
            color_mask[mask_color > 0] = (0, 255, 255)  # Yellow color

            blended = cv2.addWeighted(plain_image, 0.7, color_mask, 0.3, 0)
            cv2.rectangle(blended, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

            cv2.imshow('result', blended)
            cv2.waitKey(10)

            # #plot with matplotlib
            # plot.imshow(left_rgb)
            # plot.imshow(mask_refined, alpha=0.5)
            # #bounding box
            # x=[box[0], box[2], box[2], box[0], box[0]]
            # y=[box[1], box[1], box[3], box[3], box[1]]
            # plot.plot(x, y, 'g-')

            # plot.show(block=False)
            # plot.pause(0.01)

           

    zed_cam.close()


if __name__ == "__main__":
    main()