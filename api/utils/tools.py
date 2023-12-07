import numpy as np
import cv2


def img_cv_to_binary(img_cv):
    ext = '.png' if img_cv.ndim == 3 and img_cv.shape[2] == 4 else '.jpg'
    return cv2.imencode(ext, img_cv)[1].tobytes()


def img_binary_to_cv(img_binary):
    return cv2.imdecode(np.fromstring(img_binary, np.uint8), flags=cv2.IMREAD_UNCHANGED)
