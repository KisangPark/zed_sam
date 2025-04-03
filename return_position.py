"""
Return the position of object
-> receive depth image, segmentation mask to return position
    1) get the edge of object using mask
    2) find its center point & depth
    3) x & y coordinates using center point

    -> x, y, z positions returned
"""

import sys
import numpy as np
import argparse
import matplotlib.pyplot as plot
import PIL.Image

import pyzed.sl as sl
import cv2

from nanoowl.owl_predictor import (
    OwlPredictor
)
from nanosam.utils.predictor import Predictor


def main():

    """ 1. set arguments, define handy functions"""
    # 1st. define parser
    parser = argparse.ArgumentParser()
    # sam arguments
    parser.add_argument("--image_encoder", type=str, default="engines/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="engines/mobile_sam_mask_decoder.engine")
    #owl arguments
    parser.add_argument("--threshold", type=str, default="0.1,0.1")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="engines/owl_image_encoder_patch32.engine")
    args = parser.parse_args()
    # prompt
    list_prompt = ['a can']
    #threshold parse
    thresholds = args.threshold.strip("][()")
    thresholds = thresholds.split(',')
    if len(thresholds) == 1:
        thresholds = float(thresholds[0])
    else:
        thresholds = [float(x) for x in thresholds]
    # print(thresholds), list of values returned

    #simple bounding box transform & drawing function
    def bbox2points(bbox):
        points = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]]
        ])
        point_labels = np.array([2, 3])
        return points, point_labels
    

    """2. define models: OWL ViT, SAM"""
    detector = OwlPredictor(
        args.model,
        image_encoder_engine=args.image_encoder_engine
    )
    text_encodings = detector.encode_text(list_prompt)

    sam_model = Predictor(
        args.image_encoder,
        args.mask_decoder
    ) # in sam example code, it uses parser arguments but this is also possible -> not available..
    zed_cam = sl.Camera()



    """3. set configurations & runtime parameters, open camera, image size modification"""
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        # sys.argv: script arguments
        # 0: script file name
        # after this: arguments from command line
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NEURAL  # Mode: performance, neural, etc
    init.coordinate_units = sl.UNIT.MILLIMETER
    # Open the camera
    err = zed_cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed_cam.close()
        exit(1)
    # Set runtime parameters after opening the camera
    runtime_param = sl.RuntimeParameters()

    #image size get & modification
    image_size = zed_cam.get_camera_information().camera_configuration.resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare sl.Mat matrices
    left_image_mat = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_mat = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)



    """4. Get image, model forwarding (repeat) """
    while True:
        err = zed_cam.grab(runtime_param)

        if err == sl.ERROR_CODE.SUCCESS:
            # a. retrieve left and depth image!
            zed_cam.retrieve_image(left_image_mat, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed_cam.retrieve_image(depth_image_mat, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)

            # b. get image matrix from retrieved data
            left_image = left_image_mat.get_data()
            left_rgb = PIL.Image.fromarray(left_image).convert("RGB") #use PIL to feed model
            depth_image = depth_image_mat.get_data()
            #print("depth image:", depth_image)
            #print("left image:", left_image)


            # sam model set image
            sam_model.set_image(left_rgb)

            # c. owl vit forwarding, get bounding box
            output = detector.predict(
                image=left_rgb, 
                text=list_prompt, 
                text_encodings=text_encodings,
                threshold=thresholds,
                pad_square=False
            )
            
            N = len(output.labels)
            if N>1:
                print("Multiple cans detected, exiting...")
                #break #if N>1, multiple cans
                pass

            elif N == 0:
                print("no can detected")
                pass
            
            else:
                owl_box = output.boxes[0]
                bbox = np.array([int(x) for x in owl_box])

                points, point_labels = bbox2points(bbox)

                # d. sam forwarding: predict and return mask
                # here, should load bounding box points and labels 
                mask, _, _ = sam_model.predict(points, point_labels)

                # d. visualize -> usually with matplotlib
                mask_refined = (mask[0, 0] > 0).detach().cpu().numpy() #already numpy array
                #print("mask form:", mask_refined)

                #now, make cv image
                plain_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
                mask_color = (mask_refined * 255).astype(np.uint8) if mask_refined.max() <= 1 else binary_mask

                color_mask = np.zeros_like(plain_image)
                color_mask[mask_color > 0] = (0, 255, 255)  # Yellow color

                blended = cv2.addWeighted(plain_image, 0.7, color_mask, 0.3, 0)
                cv2.rectangle(blended, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

                cv2.imshow('result', blended)
                cv2.waitKey(10)


                # e. masking depth image
                dimg = cv2.cvtColor(depth_image, cv2.COLOR_RGB2BGR)
                mask_depth = np.zeros_like(dimg)
                mask_depth[mask_color > 0] = (1, 1, 1)
                cut_depth = cv2.multiply(dimg, mask_depth)

                cv2.imshow('depth cat', cut_depth)
                cv2.waitKey(10)


    zed_cam.close()


if __name__ == "__main__":
    main()