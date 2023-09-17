import cv2
import numpy as np


with np.load('camera_params.npz') as file:
    mtx,dist=[file[i] for i in ['mtx','dist']]

def draw_cube(img, imgpts1,imgpts2):
    imgpts1 = np.int32(imgpts1).reshape(-1,2)
    imgpts2 = np.int32(imgpts2).reshape(-1,2)

    for i in range(4):
        img = cv2.line(img, tuple(imgpts1[i]), tuple(imgpts2[i]),(255,0,0),3)
        img = cv2.line(img, tuple(imgpts1[i]), tuple(imgpts1[(i+1)%4]),(0,255,0),3)
        img = cv2.line(img, tuple(imgpts2[i]), tuple(imgpts2[(i+1)%4]),(0,0,255),3)

    return img

cube_points1 = np.float32([[2,0,0], [2,4,0], [6,4,0], [6,0,0] ])
cube_points2 = np.float32([[2,0,-4],[2,4,-4],[6,4,-4],[6,0,-4]])

img=cv2.imread('chess.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,corners = cv2.findChessboardCorners(gray,(7,6),None)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

img_cor = img*1
img_cor = cv2.drawChessboardCorners(img_cor , (7,6), corners,ret)

cv2.imwrite('corners.png',img_cor)  

ret,rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
imgpts1, jac = cv2.projectPoints(cube_points1, rvecs, tvecs, mtx, dist)
imgpts2, jac = cv2.projectPoints(cube_points2, rvecs, tvecs, mtx, dist)

img= draw_cube(img,imgpts1.astype('uint32'),imgpts2.astype('uint32'))
cv2.imwrite('res.png',img)
test = 5                                                                 