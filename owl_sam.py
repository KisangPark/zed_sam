"""
Integration: OWL_ViT + SAM
    1) zed camera setup
    2) load ViT & SAM predictor
    3) capture -> forwarding

feature points?
    -> center point of bounding box?

"""

import sys
import numpy as np
import pyzed.sl as sl
import cv2
from nanosam.utils.predictor import Predictor


def main():
    """ 1. define camera & model """
    sam_model = Predictor(
        image_encoder="data/resnet18_image_encoder.engine",
        mask_decoder="data/mobile_sam_mask_decoder.engine"
    ) # in sam example code, it uses parser arguments but this is also possible
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
    init.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT # Mode: performance, neural, etc
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



    """3. repeat -> get image, forwarding, integration"""
    while True:
        err = zed_cam.grab(runtime_param)

        if err == sl.ERROR_CODE.SUCCESS:
            # a. retrieve left and depth image!
            zed_cam.retrieve_image(left_image_mat, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed_cam.retrieve_image(depth_image_mat, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)

            # b. get image matrix from retrieved data
            left_image = left_image_mat.get_data()
            depth_image = depth_image_mat.get_data()

            # c. model forwarding: set image, predict and return mask
            sam_model.set_image(left_image)
            
            # here, should load bounding box points and labels 
            mask, _, _ = sam_model.predict(np.array([[x,y]]), np.array([1]))

            # results: depth_image & mask
            # visualize -> usually with matplotlib
            cv2.imshow('mask', mask)
            cv2.imshow('depth', depth_image)
            cv2.waitKey(10)


    cv2.destroyAllWindows()
    zed.close()


if __name__ == "__main__":
    main()