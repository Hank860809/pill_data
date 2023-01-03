import cv2
import numpy

def SIFT(ImagePath):
    img = cv2.imread(ImagePath)
    cv2.imshow('Input Image', img)
    cv2.waitKey(0)

    # 检测
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(img, None)

    # 显示
    # 必须要先初始化img2
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(0, 255, 0))
    cv2.imshow('Detected SIFT keypoints', img2)
    cv2.waitKey(0)


def SURF(ImagePath):
    img = cv2.imread(ImagePath)
    cv2.imshow('Input Image', img)
    cv2.waitKey(0)

    # 检测
    surf = cv2.xfeatures2d.SURF_create()
    keypoints = surf.detect(img, None)

    # 显示
    # 必须要先初始化img2
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(0, 255, 0))
    cv2.imshow('Detected SURF keypoints', img2)
    cv2.waitKey(0)


def ORB(ImagePath):
    img = cv2.imread(ImagePath)
    cv2.imshow('Input Image', img)
    cv2.waitKey(0)

    # 检测
    orb = cv2.ORB_create()
    keypoints = orb.detect(img, None)

    # 显示
    # 必须要先初始化img2
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(0, 255, 0))
    cv2.imshow('Detected ORB keypoints', img2)
    cv2.waitKey(0)


def AKAZE(ImagePath):
    img = cv2.imread(ImagePath)
    cv2.imshow('Input Image', img)
    cv2.waitKey(0)

    # 检测
    akaze = cv2.AKAZE_create()
    keypoints = akaze.detect(img, None)

    # 显示
    # 必须要先初始化img2
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(0, 255, 0))
    cv2.imshow('Detected AKAZE keypoints', img2)
    cv2.waitKey(0)

if __name__ == '__main__':
    ORB('pill_dataset/dataset/banch_1/175_18.png')