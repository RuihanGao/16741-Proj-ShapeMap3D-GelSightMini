from cv2 import aruco 
import cv2
import numpy as np 

mtx = np.array([[609.9723510742188, 0.0, 638.0286865234375], [ 0.0, 609.8794555664062, 367.78936767578125], [0.000000, 0.000000, 1.000000]])
dist = np.array([[0.672352], [-2.84749], [0.000455463], [0.000101013], [1.59455]])

# Jul 23, 2021 
# [[ 0.97897595  0.0580798   0.19553214 -0.03776574]
#  [-0.11820099 -0.61971325  0.77587629 -0.19906344]
#  [ 0.1662366  -0.78267632 -0.59981928  0.71964837]
#  [ 0.          0.          0.          1.        ]]

def getArucoTransform(img): 
    #Converting to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Dictionary from which Aruco Marker is generated
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

    #Detector parameters can be set here 
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    #Lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    #Making sure the no. of Id's > 0
    if np.all(ids != None):

        #Estimate the pose and get the rotational and translational vectors
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.050, mtx, dist)

        #Reshaping
        rvecs = rvec[0][0]
        tvecs = tvec[0][0]
        
        R = np.zeros((3, 3))
        cv2.Rodrigues(rvec[0], R)
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3:4] = tvec[0].T
        
        return T

