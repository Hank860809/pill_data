# #threshold
import cv2
import numpy as np

# # cv.THRESH_BINARY
# # cv.THRESH_BINARY_INV
# # cv.THRESH_TRUNC
# # cv.THRESH_TOZERO
# # cv.THRESH_TOZERO_INV
# def Thresholding(pos):
#     retval, img_bin = cv2.threshold(img, pos, 255, cv2.THRESH_BINARY_INV)
#     cv2.imshow('Binary', img_bin)

def RemoveReflections(ImgPath):
    img = cv2.imread(ImgPath, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(ImgPath)
    retval, img_bin = cv2.threshold(img, 235, 255, cv2.THRESH_BINARY_INV)
    cv2.namedWindow('Binary')
    # cv2.createTrackbar('threshold', 'Binary', 215, 255, Thresholding)
    # img_bin_a = cv2.cvtColor(img_bin, cv2.THRESH_BINARY)
    # cv2.imshow('Input', img)
    # cv2.imshow('Binary', img_bin)

    h = img_bin.shape[0]           # 取得圖片高度
    w = img_bin.shape[1]           # 取得圖片寬度
    Reflections =[]

    for x in range(w):
        for y in range(h):
            if(img_bin[y,x] == 0):
                img2[y, x] = img_bin[y, x]   # 如果在範圍內的顏色，換成背景圖的像素值
                Reflections.append((x,y))
    add_Reflections = []
    # 擴增反光區域
    # for location in Reflections:
    #     x1 = location[0]
    #     y1 = location[1]
    #     for i in range(-2,3):
    #         # add.append((x1 + i, y1))
    #         # add.append((x1, y1 + i))
    #         if (x1 + i, y1) not in add_Reflections and (x1 + i)>=0:
    #             add_Reflections.append((x1 + i, y1))
    #         if (x1, y1 + i) not in add_Reflections and (y1 + i)>=0:
    #             add_Reflections.append((x1, y1 + i))
    #     # print(len(add_Reflections))
        # exit()
    cv2.imshow('aa', img2)
    cv2.imwrite('output.png', img2)
    # cv2.imshow('a', img_bin_a)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return img2,add_Reflections

def ORB_Feature(img1,img2,ReflectionsPoint1,ReflectionsPoint2):
    # 初始化ORB
    orb = cv2.ORB_create()
    brisk = cv2.BRISK_create()
    akaze = cv2.AKAZE_create()

    Feature_model = brisk
    # 寻找关键点
    kp1 = Feature_model.detect(img1)
    kp2 = Feature_model.detect(img2)

    points1 = cv2.KeyPoint_convert(kp1)
    points2 = cv2.KeyPoint_convert(kp2)

    feature_point1 = []
    feature_point2 = []

    index1 = 0
    index2 = 0
    for i in points1:
        x = int(i[0])
        y = int(i[1])
        if (x,y) not in ReflectionsPoint1:
            feature_point1.append(kp1[index1])
        index1 +=1

    for j in points2:
        x = int(j[0])
        y = int(j[1])
        if (x,y) not in ReflectionsPoint2:
            feature_point2.append(kp2[index2])
        index2 +=1

    # 计算描述符
    kp1, des1 = Feature_model.compute(img1, feature_point1)
    kp2, des2 = Feature_model.compute(img2, feature_point2)

    # 画出关键点
    outimg1 = cv2.drawKeypoints(img1, keypoints=kp1, outImage=None)
    outimg2 = cv2.drawKeypoints(img2, keypoints=kp2, outImage=None)
    # 显示关键点
    outimg3 = np.hstack([outimg1, outimg2])
    cv2.imshow("Key Points", outimg3)
    cv2.waitKey(0)

    # 初始化 BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 对描述子进行匹配
    matches = bf.match(des1, des2)

    # 计算最大距离和最小距离
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance
    # 筛选匹配点
    '''
        当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
        但有时候最小距离会非常小，所以设置一个经验值30作为下限。
    '''
    good_match = []
    for x in matches:
        if x.distance <= max(3 * min_distance, 30):
            good_match.append(x)

    print(len(good_match))
    if len(good_match) > 20:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
        homo, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - {}/{}".format(len(good_match), 20))
        matchesMask = None

    draw_params = dict(singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_match,draw_params)
    im_out = cv2.warpPerspective(img1, homo, (640, 480))
    cv2.imshow("Match Result0", im_out)
    cv2.waitKey(0)

    h = im_out.shape[0]           # 取得圖片高度
    w = im_out.shape[1]           # 取得圖片寬度
    for x in range(w):
        for y in range(h):
            if (im_out[y,x][0] == 0):
                im_out[y,x] = img2[y,x]

    outimg4 = np.hstack([outimg1, im_out])
    cv2.imshow("Key Points 2", outimg4)
    cv2.waitKey(0)

    return homo


def draw_match(img1, img2, kp1, kp2, match, draw_params):
    outimage = cv2.drawMatches(img1, kp1, img2, kp2, match, outImg=None,**draw_params)
    cv2.imshow("Match Result", outimage)
    cv2.waitKey(0)

def show_xy(event, x, y, flags, param):
    if event == 1:
        lo = homo @ np.array([[x], [y] ,[1]])
        # print(homo)
        # print(lo)
        img2 = img.copy()  # 複製原本的圖片
        cv2.circle(img2, (x, y), 3, (0, 0, 255), -1)  # 繪製紅色圓
        cv2.circle(img2, (int(lo[0,0])+640, int(lo[1,0])), 3, (0, 0, 255), -1)  # 繪製紅色圓
        cv2.imshow('drawPoint', img2)
        # print(color)  # 印出顏色

def homography(img1, img2, visualize=False):
    FLANN_INDEX_KDTREE = 0
    SCH_PARAM_CHECKS = 50
    INDEX_PARAM_TREES = 5

    GOOD_MATCH_THRESHOLD = 0.6
    MIN_MATCH_COUNT = 10

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    # 画出关键点
    outimg1 = cv2.drawKeypoints(img1, keypoints=kp1, outImage=None)
    outimg2 = cv2.drawKeypoints(img2, keypoints=kp2, outImage=None)
    # 显示关键点
    outimg3 = np.hstack([outimg1, outimg2])
    cv2.imshow("Key Points", outimg3)
    cv2.waitKey(0)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=INDEX_PARAM_TREES)
    sch_params = dict(checks=SCH_PARAM_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, sch_params)

    matches = flann.knnMatch(desc1, desc2, k=2)
    # select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < GOOD_MATCH_THRESHOLD * n.distance:
            good_matches.append(m)
    print(len(good_matches))

    if len(good_matches) < MIN_MATCH_COUNT:
        raise (Exception('Not enough matches found'))

    src_pts = [kp1[m.queryIdx].pt for m in good_matches]
    src_pts = np.array(src_pts, dtype=np.float32).reshape((-1, 1, 2))
    dst_pts = [kp2[m.trainIdx].pt for m in good_matches]
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape((-1, 1, 2))

    homo, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
    matchesMask = mask.ravel().tolist()

    draw_params = dict(singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_matches, draw_params)
    im_out = cv2.warpPerspective(img1, homo, (640, 480))
    cv2.imshow("Match Result0", im_out)
    cv2.waitKey(0)

    im_out = RemoveReflections2(im_out)
    h = im_out.shape[0]  # 取得圖片高度
    w = im_out.shape[1]  # 取得圖片寬度
    for x in range(w):
        for y in range(h):
            if (im_out[y, x][0] == 0):
                im_out[y, x] = img2[y, x]

    outimg4 = np.hstack([outimg1, im_out])
    cv2.imshow("Key Points 2", outimg4)
    cv2.imwrite('pill_dataset/test_20221228/result0.png', im_out)
    cv2.waitKey(0)

    return homo

def sharpen(img, sigma=60):

    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

    return usm


def RemoveReflections2(Img):
    img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    retval, img_bin = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
    cv2.namedWindow('Binary')
    # cv2.createTrackbar('threshold', 'Binary', 215, 255, Thresholding)
    # img_bin_a = cv2.cvtColor(img_bin, cv2.THRESH_BINARY)
    # cv2.imshow('Input', img)
    # cv2.imshow('Binary', img_bin)

    h = img_bin.shape[0]           # 取得圖片高度
    w = img_bin.shape[1]           # 取得圖片寬度
    Reflections =[]

    for x in range(w):
        for y in range(h):
            if(img_bin[y,x] == 0):
                Img[y, x] = img_bin[y, x]   # 如果在範圍內的顏色，換成背景圖的像素值

    cv2.imshow("Img", Img)
    return Img

if __name__ == '__main__':
    # 读取图片found
    image1 = cv2.imread('pill_dataset/test_20221228/test_22.png')
    image2 = cv2.imread('pill_dataset/test_20221228/test_26.png')
    # image1,ReflectionsPoint1 = RemoveReflections('pill_dataset/test_20221214/test_16.png')
    # image2,ReflectionsPoint2 = RemoveReflections('pill_dataset/test_20221214/test_18.png')
    # homo = ORB_Feature(image1,image2,ReflectionsPoint1,ReflectionsPoint2)
    image1 = sharpen(image1)
    image2 = sharpen(image2)
    homo = homography(image1,image2)

    img = np.hstack([image1, image2])

    # print(type(M))

    cv2.imshow('drawPoint', img)
    cv2.setMouseCallback('drawPoint', show_xy)

    cv2.waitKey(0)
    cv2.destroyAllWindows()