import cv2
import numpy as np

def remove_black_borders(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

def feature_matching(img1,keypoints_1, descriptors_1, img2 ,keypoints_2, descriptors_2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1,descriptors_2,2)
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])


    img3 = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, good[:50], img2, flags=2)

    '''bf = cv2.BFMatcher()

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], None, flags=2)'''
    return good,img3

img1 = cv2.imread('1.png')
img2 = cv2.imread('2.png')
img3 = cv2.imread('3.png')
img4 = cv2.imread('4.png')
img5 = cv2.imread('5.png')
img6 = cv2.imread('6.png')


gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#sift
sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(gray1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(gray2,None)
_,sift_res = feature_matching(img1,keypoints_1, descriptors_1, img2 ,keypoints_2, descriptors_2)
cv2.imwrite('sift_res.png',sift_res)

#orb
orb = cv2.ORB_create()
keypoints_1, descriptors_1 = orb.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = orb.detectAndCompute(img2,None)
good,orb_res = feature_matching(img1,keypoints_1, descriptors_1, img2 ,keypoints_2, descriptors_2)
cv2.imwrite('orb_res.png',orb_res)

#akaze
akaze = cv2.AKAZE_create()
keypoints_1, descriptors_1 = akaze.detectAndCompute(gray1,None)
keypoints_2, descriptors_2 = akaze.detectAndCompute(gray2,None)
good,akaze_res = feature_matching(img1,keypoints_1, descriptors_1, img2 ,keypoints_2, descriptors_2)
cv2.imwrite('akaze_res12.png',akaze_res)
srcPoints = np.float32([keypoints_2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
dstPoints = np.float32([keypoints_1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC,6.0)
res12 = cv2.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
res12[0:img1.shape[0], 0:img1.shape[1]] = img1
res12 = remove_black_borders(res12)
cv2.imwrite('res12.png',res12)



keypoints_1, descriptors_1 = akaze.detectAndCompute(img3,None)
keypoints_2, descriptors_2 = akaze.detectAndCompute(img4,None)
good,orb_res = feature_matching(img3,keypoints_1, descriptors_1, img4 ,keypoints_2, descriptors_2)
cv2.imwrite('akaze_res34.png',orb_res)
srcPoints = np.float32([keypoints_2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
dstPoints = np.float32([keypoints_1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC,6.0)
res34 = cv2.warpPerspective(img4, M, (img3.shape[1] ,img4.shape[0]+img3.shape[0]  ))
res34[0:img3.shape[0], 0:img3.shape[1]] = img3
res34 = remove_black_borders(res34)
cv2.imwrite('res34.png',res34)

keypoints_1, descriptors_1 = akaze.detectAndCompute(img5,None)
keypoints_2, descriptors_2 = akaze.detectAndCompute(img6,None)
good,akaze_res = feature_matching(img5,keypoints_1, descriptors_1, img6 ,keypoints_2, descriptors_2)
cv2.imwrite('akaze_res56.png',akaze_res)
srcPoints = np.float32([keypoints_2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
dstPoints = np.float32([keypoints_1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC,6.0)
res56 = cv2.warpPerspective(img6, M, (img5.shape[1]+img6.shape[1], img6.shape[0]  ))
res56[0:img5.shape[0], 0:img5.shape[1]] = img5
res56 = remove_black_borders(res56)
cv2.imwrite('res56.png',res56)


keypoints_1, descriptors_1 = akaze.detectAndCompute(res12,None)
keypoints_2, descriptors_2 = akaze.detectAndCompute(res56,None)
good,akaze_res = feature_matching(res12,keypoints_1, descriptors_1, res56 ,keypoints_2, descriptors_2)
cv2.imwrite('akaze_res1256.png',akaze_res)
srcPoints = np.float32([keypoints_2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
dstPoints = np.float32([keypoints_1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC,4.0)
res1256 = cv2.warpPerspective(res56, M, (res34.shape[1] + res56.shape[1], res34.shape[0]))
res1256[0:res12.shape[0], 0:res12.shape[1]] = res12
res1256 = remove_black_borders(res1256)
cv2.imwrite('res1256.png',res1256)


keypoints_1, descriptors_1 = akaze.detectAndCompute(res1256,None)
keypoints_2, descriptors_2 = akaze.detectAndCompute(res34,None)
good,akaze_res = feature_matching(res1256,keypoints_1, descriptors_1, res34 ,keypoints_2, descriptors_2)
cv2.imwrite('akaze_res.png',akaze_res)
srcPoints = np.float32([keypoints_2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
dstPoints = np.float32([keypoints_1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(dstPoints, srcPoints, cv2.RANSAC,6.0)
res = cv2.warpPerspective(res1256, M, ( res34.shape[1]+res1256.shape[1],res34.shape[0] ))
res[0:res34.shape[0], 0:res34.shape[1]] = res34
res = remove_black_borders(res)
cv2.imwrite('res.png',res)

test=5