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
import argparse
import matplotlib.pyplot as plot
import PIL.Image

import pyzed.sl as sl

from nanosam.utils.owlvit import OwlVit
from nanosam.utils.predictor import Predictor


def main():

    """ 1. set arguments, define handy functions"""
    # 1st. define parser
    parser = argparse.ArgumentParser()
    # sam arguments
    parser.add_argument("--image_encoder", type=str, default="engines/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="engines/mobile_sam_mask_decoder.engine")
    #owl arguments
    parser.add_argument("--prompt", nargs='+', type=str, default="can")
    parser.add_argument("--thresh", type=float, default=0.1)
    args = parser.parse_args()

    #simple bounding box transform & drawing function
    def bbox2points(bbox):
        points = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]]
        ])
        point_labels = np.array([2, 3])
        return points, point_labels
    
    def draw_bbox(bbox):
        x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
        y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
        plot.plot(x, y, 'g-')




    """2. define models: OWL ViT, SAM"""
    detector = OwlVit(args.thresh)

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

            # sam model set image
            sam_model.set_image(left_rgb)

            # c. owl vit forwarding, get bounding box
            detections = detector.predict(left_rgb, texts=args.prompt)
            N = len(detections)
            if N>1:
                print("Multiple cans detected, exiting...")
                break #if N>1, multiple cans

            elif N == 0:
                plot.imshow(left_rgb)
                plot.show(block=False)
                plot.pause(0.01)
            
            else:
                bbox = detections[0]['bbox']
                points, point_labels = bbox2points(bbox)

                # d. sam forwarding: predict and return mask
                # here, should load bounding box points and labels 
                mask, _, _ = sam_model.predict(points, point_labels)

                # d. visualize -> usually with matplotlib
                mask_refined = (mask[0, 0] > 0).detach().cpu().numpy()

                #plot with matplotlib
                plot.imshow(left_rgb)
                plot.imshow(mask_refined, alpha=0.5)
                #bounding box
                draw_bbox(bbox)
                plot.show(block=False)
                plot.pause(0.01)
            

    zed_cam.close()


if __name__ == "__main__":
    main()