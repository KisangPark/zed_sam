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
        
        # threshold parse
        self.thresholds = args.threshold.strip("][()")
        self.thresholds = self.thresholds.split(',')
        if len(self.thresholds) == 1:
            self.thresholds = float(self.thresholds[0])
        else:
            self.thresholds = [float(x) for x in self.thresholds]
        
        """2. define models: OWL ViT, SAM"""
        self.detector = OwlPredictor(
            args.model,
            image_encoder_engine=args.image_encoder_engine
        )
        
        self.sam_model = Predictor(
            args.image_encoder,
            args.mask_decoder
        ) # in sam example code, it uses parser arguments but this is also possible -> not available..


    def get_mask(self, text_prompt, image): # use owl & sam to get object mask
        # text prompt encoding
        try:
            list_prompt = [text_prompt]
        except:
            list_prompt = ['a can']
        
        text_encodings = self.detector.encode_text(list_prompt)
        self.sam_model.set_image(image)

        output = self.detector.predict(
                image=image, 
                text = list_prompt, 
                text_encodings=text_encodings,
                threshold=self.thresholds,
                pad_square=False
            )

        N = len(output.labels)
        if N>1:
            print("Multiple cans detected, pausing...")
            pass #if N>1, multiple cans

        elif N == 0:
            print("no can detected")
            pass
        
        else:    
            owl_box = output.boxes[0]
            bbox = np.array([int(x) for x in owl_box])

            points, point_labels = bbox2points(bbox)

            # d. sam forwarding: predict and return mask
            # here, should load bounding box points and labels 
            mask, _, _ = self.sam_model.predict(points, point_labels)

            # d. mask to numpy array
            mask_refined = (mask[0, 0] > 0).detach().cpu().numpy() #already numpy array

            #now, make cv image
            plain_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
            mask_color = (mask_refined * 255).astype(np.uint8) if mask_refined.max() <= 1 else binary_mask

            color_mask = np.zeros_like(plain_image)
            color_mask[mask_color > 0] = (0, 255, 255)  # Yellow color

            blended = cv2.addWeighted(plain_image, 0.7, color_mask, 0.3, 0)
            cv2.rectangle(blended, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

            # here: blended: image, mask_refined: boolean mask

        return mask, blended


    def return_2d_center(self, mask):
        # boolean to int
        mask = mask.astype("uint8")

        #mesh matrix: represents the coordinate of each pixel
        mesh_y, mesh_x = np.mgrid[0:mask.shape[0]:1, 0:mask.shape[1]:1]

        # bitwise AND between mesh and mask
        if np.count_nonzero(mask):
            centroid_x = int(float(np.sum(cv2.bitwise_and(mesh_x, mesh_x, mask=mask))/np.count_nonzero(mask)))
            print("x coordinate:", centroid_x)
            centroid_y = int(float(np.sum(cv2.bitwise_and(mesh_y, mesh_y, mask=mask)))/np.count_nonzero(mask))
            print("y coordinate:", centroid_y)
            #by using bitwise AND, only true-masked pixels alive -> sum them all!
        else:
            centroid_x = mask.shape[1] /2
            centroid_y = mask.shape[0] /2
            # if no mask, center of picture is COM

        return centroid_x, centroid_y


    def return_3d_center(self, pixel_center, depth):
        # get 2d center, matrix to make 3d center value

        matrix = np.open(matrix.npy)
        np.linalg