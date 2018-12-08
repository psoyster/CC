# -----------------------------------------------------------------------------#
# Authors: Paul Soyster, Tyler Loukus
# Created on 11/12/2018
# Program: cc_main.py
# Project: Car Counting for parking lot monitoring
# VCS: https://github.com/psoyster/CC
#
# Purpose:
#     - Evaluate a moving objects and if it is a car, track the direction it is
#       traveling.
#     - If traveling into the parking lot, count +1 cars in the parking lot.
#     - If traveling out of the parking lot, count -1 cars in the parking lot.
#     - Use this data to extrapolate the number of parking spaces that should
#       be unoccupied in the parking lot.
#
#
# Updated: 12/7/2018
# -----------------------------------------------------------------------------#


import cv2
import numpy as np

# cars = cv2.CascadeClassifier('cars.xml')

count_back = 0
count_forward = 0
total = 0
ct = 0

config = {'thresh_min': 700,
          'thresh_max': 12000,
          'save_video': False
          }

# video file used for evaluation.
vid = r"two_way.mp4"
vid_out = r"vidout_bw_.mp4"
cap = cv2.VideoCapture(vid)

# Info pertaining to the video file
video_info = {'fps': cap.get(cv2.CAP_PROP_FPS),
              'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
              'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
              }

if config['save_video']:
    fourcc = cv2.VideoWriter_fourcc('C', 'R', 'A', 'M')
    # CRAM is video format for Microsoft Video1

    # format the saved video with the vid_out image
    out = cv2.VideoWriter(vid_out, -1, 25.0,  # video_info['fps'],
                          (video_info['width'], video_info['height']))




################################################################################

'''Background Subtraction Methods:

    MOG2: Gaussian mixture background subtractor 
        - history: Number of frames to compare to
        - varThreshold: the squared Mahalanobis distance between the pixel 
          and the model.  (Usually (4)**2)
        - detectShadows: If true, will return detected shadow pixels as grey 
          but will slow down processing


    KNN: K-Nearest Neighbors
        - history: Number of frames to compare to
        - dist2Threshold: Squared distance between pixel and the sample 
        - detectShadows: If true, will return detected shadow pixels as grey 
          but will slow down processing
'''

# MOG2
# fgbg = cv2.createBackgroundSubtractorMOG2(history=100,
#                                           varThreshold=16,
#                                           detectShadows=True)

# KNN
fgbg = cv2.createBackgroundSubtractorKNN(history=200,
                                         dist2Threshold=100.0,
                                         detectShadows=False)

################################################################################





cap.set(cv2.CAP_PROP_POS_FRAMES, 10)  # jump to frame

while (cap.isOpened()):
    ret, frame = cap.read()
    if (ret != True):  # if unable to open the video
        cap.release()  # release the video feed
        print("ret = False")
        break

    # frame_out will be used to display the functions being acted upon the frame
    frame_out = frame.copy()

    # black and gray version of the video
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Blurring the gray frame with 3x3 Gaussian kernel
    frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 3, 3, cv2.BORDER_DEFAULT)

    # Applying the blurred image to the background subtraction result image
    fgmask = fgbg.apply(frame_blur)





##########################  Thresholds  ########################################

    # mov = np.uint8(fgmask > 200) * 255

    # mov = np.uint8(fgmask)

    _, mov = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)

    # mov = cv2.adaptiveThreshold(mov, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 51, 20)

################################################################################





################################################################################

    '''Image Filters

    kernel: 2D matrix with configurable size and value for the image convolution

    closing: Dilation followed by erosion 
        - Good for closing small holes in the foreground image

    opening: Erosion followed by dilation
        - Good for removing noise in image

    gradient: The difference between opening and closing
        - Produces the outline of the object 
    '''

    # --------------------------------------------------------------------------
    # Varying sized kernel
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    # --------------------------------------------------------------------------
    # Fixed sized kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    closing = cv2.morphologyEx(mov, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(mov, cv2.MORPH_OPEN, kernel)
    gradient = cv2.morphologyEx(mov, cv2.MORPH_GRADIENT, kernel)
    # --------------------------------------------------------------------------
    # Custom value kernel
    edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    closing2 = cv2.morphologyEx(mov, cv2.MORPH_CLOSE, edge_kernel)
    opening2 = cv2.morphologyEx(mov, cv2.MORPH_OPEN, edge_kernel)
    gradient2 = cv2.morphologyEx(mov, cv2.MORPH_GRADIENT, edge_kernel)

################################################################################





# ----------------------  Filter Selection  -----------------------------------#

    # mov = cv2.erode(mov, erode, iterations=1)
    # mov = cv2.dilate(mov, dilate, iterations=1)
    # mov = cv2.dilate(mov, dilate2, iterations=1)

    # mov = closing
    # mov = opening
    # mov = gradient

    # mov = closing2
    # mov = opening2
    # mov = gradient2

# -----------------------------------------------------------------------------#





# ------------  Marking Vehicle's Perimeter and Center Point  -----------------#

    # mode = external two points of the detected contour,
    # method = uses only the outer most four x,y points
    # both are used as aproximations of the contour boxes for speed
    (_, vehicle, hierarchy) = cv2.findContours(image=mov,
                                               mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_SIMPLE)



    for edge in vehicle:
        # print(edge)
        veh_area = cv2.contourArea(edge)
        if (veh_area < config['thresh_min']) or \
                (veh_area > config['thresh_max']):
            # print(veh_area)
            continue

        # x,y starting point and width, height extensions of rectangle
        (x, y, w, h) = cv2.boundingRect(edge)


        # Drawing a blue rectangle around the detected vehicle
        # cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.rectangle(mov, (x, y), (x + w, y + h), (255, 0, 0), 2)


        # Drawing rectangles around objects with different colors based on 
        # their hierarchy
        for i in range(-1, 10):
            if hierarchy.any() == i:
                cv2.rectangle(frame_out, (x, y), (x + w, y + h),
                              (i*25, i*10,i*5), 2)

                cv2.rectangle(mov, (x, y), (x + w, y + h),
                              (i*25, i*10, i*5), 2)




        # Finding the centroid
        M = cv2.moments(edge)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Dot at the center of the vehicle blob
        cv2.circle(frame_out, (cX, cY), 5, (255, 255, 255), -1)
# -----------------------------------------------------------------------------#





# -----------------------------------------------------------------------------#
        # Direction based on location of centroid in relation to the
        # evaluation area of both directions of traffic

        # Checking if the centroid has passed the first line
        if (cY >= 399 and cY <= 401) and (cX >= 400 and cX <= 600):
            count_forward += 1
            ct += 1

        if (cY >= 399 and cY <= 401) and (cX >= 675 and cX <= 1000):
            count_back += 1
            ct -= 1

        # Number of cars in the lot cannot be negative
        if ct < 0: ct = 0

# -----------------------------------------------------------------------------#


    # Evaluation lines
    yellow = cv2.line(frame_out, (300, 400), (950, 400), (0, 255, 255), 2)
    red = cv2.line(frame_out, (300, 410), (950, 410), (0, 0, 255), 2)



    # Print how many cars have been counted in the left lanes in green
    cv2.putText(frame_out, ("%d" % count_forward), (205, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Print cars traveling in the right lanes in red with added white so its
    # easier to see
    cv2.putText(frame_out, ("%d" % count_back), (1005, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.putText(frame_out, ("%d" % count_back), (1006, 451),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(frame_out, ("%d" % count_back), (1005, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)


    # Print the count of vehicles in the lot based on cars in vs cars going out
    cv2.putText(frame_out, ("%d" % ct), (620, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    #  Print the current frame, top right corner
    cv2.putText(frame_out, ("%d" % current_frame), (1100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)



    if config['save_video']:
        out.write(frame_out)  # save the color video
        # out.write(mov)        #  save the background subtraction video



    cv2.imshow('frame_out', frame_out)
    # cv2.imshow('mov', mov)
    # cv2.imshow('fgmask', fgmask)



    # Pressing 'q' key will close the video, end the program normally
    k = cv2.waitKey(5)
    if k == ord('q'):
        break







cap.release()
if config['save_video']:
    out.release()
cv2.destroyAllWindows()


