"""
Inegrated code
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


# bounding box function
def bbox2points(bbox):
    points = np.array([
        [bbox[0], bbox[1]],
        [bbox[2], bbox[3]]
    ])
    point_labels = np.array([2, 3])
    return points, point_labels


class OBJECT_LOCATOR():

    def __init__(self, args): #get arguments for models, get text prompt
        
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
        self.detector = OwlPredictor(
            args.model,
            image_encoder_engine=args.image_encoder_engine
        )
        
        self.sam_model = Predictor(
            args.image_encoder,
            args.mask_decoder
        ) # in sam example code, it uses parser arguments but this is also possible -> not available..


    def get_mask(self, image):
        #pil image
        pil_image = PIL.Image.fromarray(image).convert("RGB")
        
        # text prompt encoding
        text_encodings = self.detector.encode_text(self.prompt)
        
        # set sam image & owl vit detection
        self.sam_model.set_image(pil_image)

        output = self.detector.predict(
                image=pil_image, 
                text = self.prompt, 
                text_encodings=text_encodings,
                threshold=self.thresholds,
                pad_square=False
            )

        """detect single can"""
        N = len(output.labels)
        if N>1:
            print("Multiple cans detected, passing...")
            return None, image

        elif N == 0:
            print("no can detected, passing")
            return None, image
        
            """draw bounding box, segmentation, return mask"""
        else:    
            owl_box = output.boxes[0]
            bbox = np.array([int(x) for x in owl_box])

            points, point_labels = bbox2points(bbox)
            mask, _, _ = self.sam_model.predict(points, point_labels)
            mask_refined = (mask[0, 0] > 0).detach().cpu().numpy() #already numpy array

            # make blended image
            plain_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            plain_image = cv2.cvtColor(plain_image, cv2.COLOR_BGR2RGB)
            mask_color = (mask_refined * 255).astype(np.uint8) if mask_refined.max() <= 1 else binary_mask

            color_mask = np.zeros_like(plain_image)
            color_mask[mask_color > 0] = (0, 255, 255)  # Yellow color

            blended = cv2.addWeighted(plain_image, 0.7, color_mask, 0.3, 0)
            cv2.rectangle(blended, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

            return mask_refined, blended


    def return_2d_center(self, mask):
        # boolean to int
        mask = mask.astype("uint8")

        #mesh matrix: represents the coordinate of each pixel
        mesh_y, mesh_x = np.mgrid[0:mask.shape[0]:1, 0:mask.shape[1]:1]

        # bitwise AND between mesh and mask
        if np.count_nonzero(mask):
            centroid_x = int(float(np.sum(cv2.bitwise_and(mesh_x, mesh_x, mask=mask))/np.count_nonzero(mask)))
            centroid_y = int(float(np.sum(cv2.bitwise_and(mesh_y, mesh_y, mask=mask)))/np.count_nonzero(mask))
            #by using bitwise AND, only true-masked pixels alive -> sum them all!
        else:
            centroid_x = mask.shape[1] /2
            centroid_y = mask.shape[0] /2
            # if no mask, center of picture is COM

        return np.array([centroid_x, centroid_y]) #return as numpy array


    def return_3d_center(self, pixel_center, depth):
        # get 2d center, matrix to make 3d center value
        center = pixel_center.tolist()
        center.append(depth)
        print(center)

        mtx = np.load("data/intrinsic_matrix.npy")
        inv_mtx = np.linalg.inv(mtx) #inverse matrix

        return np.matmul(inv_mtx, np.array(center))




"""
main function
"""

def main():
    """initialize parser, camera object"""

    # 1. parser declaration
    parser = argparse.ArgumentParser()
    # 1-1. sam arguments
    parser.add_argument("--image_encoder", type=str, default="engines/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="engines/mobile_sam_mask_decoder.engine")
    # 1-2. owl arguments
    parser.add_argument("--threshold", type=str, default="0.1,0.1")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="engines/owl_image_encoder_patch32.engine")
    parser.add_argument("--prompt", type=str, default="a hand")
    # change to arguments
    args = parser.parse_args()


    """declare locator object"""
    locator = OBJECT_LOCATOR(args)


    """initialize camera object"""
    # 2. camera object declaration & define sl matrices
    zed_cam = sl.Camera()

    # 2-1. initial parameters
    input_type = sl.InputType()
    if len(sys.argv) >=2: # script arguments, first file name after command line arguments
        input_type.set_from_svo_file(sys.argv[1]) # set type from arguments
    
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD720 # 720
    init.depth_mode = sl.DEPTH_MODE.NEURAL # modes: PERFORMANCE, NEURAL_PLUS, NEURAL_LIGHT
    init.coordinate_units = sl.UNIT.MILLIMETER

    # 2-2. open camera
    err = zed_cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed_cam.close()
        exit(1)
    # Set runtime parameters after opening the camera
    runtime_param = sl.RuntimeParameters()

    # 2-3. declare image size & sl matrices
    image_size = zed_cam.get_camera_information().camera_configuration.resolution
    # image_size.width = image_size.width /2 # half resolution
    # image_size.height = image_size.height /2

    # 2-4. Declare sl.Mat matrices
    left_image_mat = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    #depth image matrix
    depth_image_mat = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    
    depth_mat = sl.Mat()
    point_cloud_mat = sl.Mat()

    """grab image, use methods to get 3D point"""
    while True:
        err = zed_cam.grab(runtime_param)

        if err == sl.ERROR_CODE.SUCCESS:
            # 3-1. retrieve data
            zed_cam.retrieve_image(left_image_mat, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed_cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            zed_cam.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)

            # depth image
            zed_cam.retrieve_image(depth_image_mat, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)

            # 3-2. get image and mask
            image = left_image_mat.get_data()
            mask, blended_image = locator.get_mask(image)

            # depth image
            depth_image = depth_image_mat.get_data()

            # check mask existence
            if mask is None:
                # if none, show image and quit
                cv2.imshow("blended image", blended_image)
                cv2.imshow("depth image", depth_image)
                cv2.waitKey(10)
                
            else:
                # 3-3. get depth (both method and point cloud)
                # 3-3-1. get 3d point from method
                center_2d = locator.return_2d_center(mask)
                depth = depth_mat.get_value(int(center_2d[0]), int(center_2d[1]))
                # list (SUCCESS, value) returned

                # modify 2d center related to center pixel
                modified_center = center_2d - np.array([mask.shape[1]/2, mask.shape[0]/2])
                # print(f"Center X Pixel: {mask.shape[1]/2} , Center Y Pixel: {mask.shape[0]/2}")
                point_by_class = locator.return_3d_center(modified_center, depth[1])

                # 3-3-2. get 3d point from point cloud
                temp = point_cloud_mat.get_value(int(center_2d[0]), int(center_2d[1]))
                # list (SUCCESS, value) returned
                tmp = temp[1]
                point_by_cloud = (tmp[0], tmp[1], tmp[2])

                # 3-4. image show & get points
                cv2.imshow("blended image", blended_image)
                cv2.imshow("depth image", depth_image)
                cv2.waitKey(10)
                print("point by class:", point_by_class)
                print("point by cloud:", point_by_cloud)

    zed_cam.close()



if __name__ == "__main__":
    main()