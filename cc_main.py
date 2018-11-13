# -----------------------------------------------------------------------------#
# Authors: Paul Soyster, Tyler Loukus
# Created on 11/12/2018
# Program: cc_main.py
# Project: Car Counting for parking lot monitoring
# VCS: https://github.com/psoyster/CC
#
# Purpose:
#     Evaluate a moving objects and if it is a car, track the direction it is
#     traveling.  If traveling into the parking lot, count +1 cars in the
#     parking lot.  If traveling out of the parking lot, count -1 cars in the
#     parking lot.  Use this data to extrapolate the number of parking spaces
#     that should be unoccupied in the parking lot.
#
#
# Updated: 11/12/2018
# -----------------------------------------------------------------------------#


import cv2
import numpy as np

config = {'thresh_min': 2000,
          'thresh_max': 7000,
           'max_spots': 112}


# video file used for evaluation.
# need to look into how would use an IP address
vid = r"highway.mp4"
cap = cv2.VideoCapture(vid)

# Cascade xml file used for the training
# cascade_file = r""
# cars_cascade = cv2.CascadeClassifier(cascade_file)

# Info pertaining to the video file
vid_info = {'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
            'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}

# MOG2 background subtraction method.
fgbg = cv2.createBackgroundSubtractorMOG2(history=300,
                                          varThreshold=16,
                                          detectShadows=True)

cap.set(cv2.CAP_PROP_POS_FRAMES, 450)  # jump to frame
erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19))


while (cap.isOpened()):
    ret, frame = cap.read()
    if (ret != True):
        break
        cap.release()
        print("ret = False")

    frame_out = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blurring the gray frame with 3x3 Gaussian kernel
    blur = cv2.GaussianBlur(src=frame.copy(), ksize=(3, 3), sigmaX=0)
    mask = fgbg.apply(blur)

    mov = np.uint8(mask == 255) * 255
    # mov = cv2.erode(mov, erode, iterations=1)
    # mov = cv2.dilate(mov, dilate, iterations = 1)


    (_, vehicle, _) = cv2.findContours(image=mov, mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_SIMPLE)


    for edge in vehicle:
        # print(edge)
        veh_area = cv2.contourArea(edge)
        if (veh_area < config['thresh_min']) | (veh_area > config[
            'thresh_max']):
            continue
        (x, y, w, h) = cv2.boundingRect(edge)
        cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 0, 0), 2)








    cv2.imshow('frame_out', frame_out)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
