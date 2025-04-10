"""
flow
1) 


only intrinsic 
"""

"""
TODO
    1) camera calibration, get intrinsic camera parameter
    2) get z position, make 3D vector
    3) matmul: inverse intrinsic & 3D vector
    -> get 3D
"""

import time
import cv2
import numpy as np
import os
import glob
import sys
import PIL.Image

import pyzed.sl as sl



def calibration():
    # declare checkerboard info
    img_shape = []

    # size of checkerboard
    CHECKERBOARD = (6,9) #inner intersections

    #criteria for terminalization
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D points save
    points_3D = []
    # 2D points save
    points_2D = []

    # make fundamental points (target?)
    object_point = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    object_point[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    object_point *= 20 #20mm for one checkerboard pixel

    prev_img_shape = None

    #get glob images
    images = glob.glob('/home/kisangpark/lab_training/calibration/data/*.jpg')

    print(images)
    for file in images:
        img = cv2.imread(file)
        # grayscale (efficiency)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


        # finding corners
        ret, corners = cv2.findChessboardCorners(gray,
                                                CHECKERBOARD,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # image, patternsize, falgs
        #ret = true if number of corners are correct / corners return the list of corner coordinates                
        

        if ret == True:
            points_3D.append(object_point)

            # precision
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            # image, corners, window size, zerozone, criteria
            img_shape = gray.shape[::-1]
            print (img_shape)

            points_2D.append(corners2) # 2D points obtained by checkerboard image
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img',img)
        cv2.waitKey(2000)
    cv2.destroyAllWindows()

    # calibration, return matrices
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points_3D, points_2D, img_shape, None, None)
    print("intrinsic parameter:", mtx)

    return mtx



def inverse_to_3D(mtx, center_2d, depth):
    inv_mtx = np.linalg.inv(mtx)

    # make 3 - dimensional pixel vector
    center_2d.append(depth)
    # check shape

    # matmul
    result = np.matmul(inv_mtx, center_2d)

    return result




def main():
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

    # 10 images of checkerboard -> capture 10 times
    for i in len():
        err = zed_cam.grab(runtime_param)

        if err == sl.ERROR_CODE.SUCCESS:
            # a. retrieve left and depth image!
            zed_cam.retrieve_image(left_image_mat, sl.VIEW.LEFT, sl.MEM.CPU, image_size)

            # b. get image matrix from retrieved data
            left_image = left_image_mat.get_data()
            
            # visualize, save --> 5 second
            cv2.imshow("left image", left_image)
            path = os.path.join("calibration", i, ".jpg")
            cv2.imwrite(path, left_image)

            time.sleep(5)


    mtx = calibration()

    # get center 2 dimensional coordinate
    # mask
    # make upper as functions, import to main code

    # get depth -> depth api code
    depth_map = sl.Mat()
    zed_cam.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

    depth = depth_map.get_vallue(center_2d[0], center_2d[1])

    # get 3D coordinate vector
    result = inverse_to_3D(mtx, center_2d, depth)

