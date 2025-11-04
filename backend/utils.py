# backend/utils.py
import base64
import cv2
import numpy as np

def read_image_from_base64(b64_string):
    """Prend base64 sans prefix 'data:image/png;base64,'"""
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img