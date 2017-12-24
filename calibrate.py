from glob import glob
import pickle
import os

import numpy as np
import cv2


_CALIBRATION_IMAGES = glob(os.path.join("camera_cal", "calibration*.jpg"))


def calibrate_camera():
    image_size = (1280, 720)
    object_points = []
    image_points = []

    obj_point = np.zeros((9 * 6, 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for cal_image in _CALIBRATION_IMAGES:
        img = cv2.imread(cal_image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            image_points.append(corners)
            object_points.append(obj_point)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

    dist_pickle = dict()
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("camera_cal/wide_dist_pickle.p", "wb"))
    return mtx, dist


def get_camera_calibration():
    with open("camera_cal/wide_dist_pickle.p", "rb") as _file:
        data = pickle.load(_file)
        mtx = data.get("mtx", None)
        dist = data.get("dist", None)
    return mtx, dist
