# #threshold
import cv2
import numpy as np
import os
import demo_superpoint_test as sp

# # cv.THRESH_BINARY
# # cv.THRESH_BINARY_INV
# # cv.THRESH_TRUNC
# # cv.THRESH_TOZERO
# # cv.THRESH_TOZERO_INV
# def Thresholding(pos):
#     retval, img_bin = cv2.threshold(img, pos, 255, cv2.THRESH_BINARY_INV)
#     cv2.imshow('Binary', img_bin)
def Keypoint_detection(img1,img2,Feature_model):

    # 尋找關鍵點和計算描述符
    kp1, des1 = Feature_model.detectAndCompute(img1, None)
    kp2, des2 = Feature_model.detectAndCompute(img2, None)

    # 画出关键点
    outimg1 = cv2.drawKeypoints(img1, keypoints=kp1, outImage=None)
    outimg2 = cv2.drawKeypoints(img2, keypoints=kp2, outImage=None)
    # 显示关键点
    outimg3 = np.hstack([outimg1, outimg2])
    # cv2.imshow("Key Points", outimg3)
    # cv2.waitKey(0)

    return kp1,des1,kp2,des2

def Feature_matching(img1,img2,kp1,des1,kp2,des2):
    # 初始化匹配模型 BFMatcher
    bf_HAMMING = cv2.BFMatcher(cv2.NORM_HAMMING)
    bf_L2 = cv2.BFMatcher(cv2.NORM_L2)
    # 初始化匹配模型 FlannBasedMatcher
    FLANN_INDEX_KDTREE = 0
    SCH_PARAM_CHECKS = 50
    INDEX_PARAM_TREES = 5

    GOOD_MATCH_THRESHOLD = 0.6
    MIN_MATCH_COUNT = 10

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=INDEX_PARAM_TREES)
    sch_params = dict(checks=SCH_PARAM_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, sch_params)

    # Matcher 選擇 bf_HAMMING,bf_L2,flann

    Matcher = bf_HAMMING

    # 对描述子进行匹配
    # matches = Matcher.match(des1, des2)
    try:
        matches = Matcher.knnMatch(des1, des2, k=2)
    except:
        Matcher = bf_L2
        matches = Matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    try:
        # 计算最大距离和最小距离
        min_distance = matches[0].distance
        max_distance = matches[0].distance
        for x in matches:
            if x.distance < min_distance:
                min_distance = x.distance
            if x.distance > max_distance:
                max_distance = x.distance

        for x in matches:
            if x.distance <= max(3 * min_distance, 30):
                good_matches.append(x)
    except:
        for m, n in matches:
            if m.distance < GOOD_MATCH_THRESHOLD * n.distance:
                good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homo, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 4))
        matchesMask = None

    draw_params = dict(singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    # 绘制匹配结果
    # draw_match(img1, img2, kp1, kp2, good_matches, draw_params)

    return homo

def draw_match(img1, img2, kp1, kp2, match, draw_params):
    outimage = cv2.drawMatches(img1, kp1, img2, kp2, match, outImg=None,**draw_params)
    cv2.imshow("Match Result", outimage)
    cv2.waitKey(0)

def show_xy(event, x, y, flags, param):
    if event == 1:
        print(homo)
        print(np.array([[x], [y] ,[1]]))

        g = np.random.randint(0, 200)
        b = np.random.randint(0, 200)
        color = (b, g, 255)

        lo = homo @ np.array([[x], [y] ,[1]])
        print(lo)
        img2 = img.copy()  # 複製原本的圖片
        cv2.circle(img, (x, y), 3, color, -1)  # 繪製紅色圓
        cv2.circle(img, (int(lo[0,0])+540, int(lo[1,0])), 3, color, -1)  # 繪製紅色圓
        cv2.imshow('drawPoint', img)
        print(color)  # 印出顏色

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

    for x in range(w):
        for y in range(h):
            if(img_bin[y,x] == 0):
                Img[y, x] = img_bin[y, x]   # 如果在範圍內的顏色，換成背景圖的像素值

    # cv2.imshow("Img", Img)
    return Img

def sharpen(img, sigma=60):

    # img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, img, -0.5, 0)
    # cv2.imshow('usm', usm)

    return usm

def Binary(Img):
    img =cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    img2 = Img
    retval, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

    h = img_bin.shape[0]           # 取得圖片高度
    w = img_bin.shape[1]           # 取得圖片寬度

    cv2.imwrite('output.png', img2)

    return img_bin

# 調整兩張照片亮度
def adjust_brightness(Img1,Img2):
    # 將兩張照片轉換為灰度圖像
    Img1_gary =cv2.cvtColor(Img1, cv2.COLOR_BGR2GRAY)
    Img2_gary =cv2.cvtColor(Img2, cv2.COLOR_BGR2GRAY)
    # 將灰度圖像轉換為浮點數類型
    Img1_gary = Img1_gary.astype(float)
    Img2_gary = Img2_gary.astype(float)

    # 計算兩張照片的均值和標準差
    mean1, std1 = cv2.meanStdDev(Img1_gary)
    mean2, std2 = cv2.meanStdDev(Img2_gary)

    # 對兩張照片進行亮度平衡
    # cv2.imshow('Img2_before', Img2)
    # Img1 = np.uint8(np.clip((Img1 - mean1) / std1 * std2 + mean2 , 0, 255))
    Img2 = np.uint8(np.clip((Img2 - mean2) / std2 * std1 + mean1, 0, 255))
    # cv2.imshow('Img2_after', Img2)

    return Img1,Img2

# 將反光部分轉換成黑色像素
def  RemoveReflections(Img1):
    tmp = Img1.copy()
    Img1_gary =cv2.cvtColor(Img1, cv2.COLOR_BGR2GRAY)

    Img1_lab = cv2.cvtColor(Img1, cv2.COLOR_BGR2LAB)
    # Define the lower and upper threshold for white color in LAB color space
    lower_threshold = np.array([248, 128, 128], dtype=np.uint8)
    upper_threshold = np.array([255, 255, 255], dtype=np.uint8)

    mask_lab = cv2.inRange(Img1_lab, lower_threshold, upper_threshold)

    highlighted_area = cv2.bitwise_and(Img1, Img1, mask=mask_lab)
    highlighted_area_gary = cv2.cvtColor(highlighted_area, cv2.COLOR_BGR2GRAY)

    retval, Img1_bin = cv2.threshold(Img1_gary, 235, 255, cv2.THRESH_BINARY)

    h = Img1_bin.shape[0]           # 取得圖片高度
    w = Img1_bin.shape[1]           # 取得圖片寬度

    cv2.imshow('RemoveReflections_mask', Img1_bin)
    cv2.imshow('highlighted_area_gary', highlighted_area_gary)
    cv2.waitKey(0)
    for x in range(w):
        for y in range(h):
            if(Img1_bin[y,x] == 255 and highlighted_area_gary[y,x] == 255):
            # if(Img1_bin[y,x] == 0):
                tmp[y, x] = 0   # 如果在範圍內的顏色，換成背景圖的像素值

    cv2.imshow('tmp', tmp)
    return tmp


def blend_images(image1, image2, mask):
    # 將遮罩轉換為浮點數型態
    # mask = mask.astype(np.float32) / 255.0

    # 進行泊松融合操作
    blended_image = cv2.seamlessClone(image1, image2, mask, (320, 240), cv2.MIXED_CLONE)

    return blended_image

def homography_image(img,homo):

    Img = cv2.warpPerspective(img, homo, (img.shape[1], img.shape[0]))

    return Img
# 合成兩張照片
def merge_images(Img1,Img2,homo):

    Img1_homo = homography_image(Img1,homo)

    # cv2.imshow('Img1_homo', Img1_homo)
    # cv2.imshow('Img2', Img2)

    # cv2.waitKey(0)
    h = Img1.shape[0]           # 取得圖片高度
    w = Img2.shape[1]           # 取得圖片寬度

    # 宣告一個和照片相同大小的陣列來存儲灰度值
    img1_gray_intensity = np.zeros((h, w), dtype=np.float32)
    img2_gray_intensity = np.zeros((h, w), dtype=np.float32)
    gray_intensity_mean = np.zeros((h, w), dtype=np.float32)
    for x in range(w):
        for y in range(h):
            Img1_pixel = Img1_homo[y, x]
            Img2_pixel = Img2[y, x]
            img1_gray_intensity[y, x] = (0.299 * Img1_pixel[0] + 0.587 * Img1_pixel[1] + 0.114 * Img1_pixel[2])
            img2_gray_intensity[y, x] = (0.299 * Img2_pixel[0] + 0.587 * Img2_pixel[1] + 0.114 * Img2_pixel[2])
            gray_intensity_mean[y, x] = ((img1_gray_intensity[y, x] + img2_gray_intensity[y, x]) * 0.5)
            if (img1_gray_intensity[y,x] == 0):
                Img1_homo[y,x] = Img2[y,x]

    # temp1,temp2,temp3分別用來顯示 閥值30 / 閥值50 / 不取閥值 合成後的照片
    temp1 = Img1_homo.copy()
    temp2 = Img1_homo.copy()
    temp3 = Img1_homo.copy()

    for x in range(w):
        for y in range(h):
            if ((img1_gray_intensity[y,x] - gray_intensity_mean[y,x]) >= 30):
                temp1[y,x] = Img1_homo[y,x]*0.5 + Img2[y,x]*0.5
            if ((img1_gray_intensity[y,x] - gray_intensity_mean[y,x]) >= 50):
                temp2[y,x] = Img1_homo[y,x]*0.5 + Img2[y,x]*0.5
            temp3[y,x] = (Img1_homo[y,x]*img2_gray_intensity[y,x]/(gray_intensity_mean[y,x]*2)+Img2[y,x]*img1_gray_intensity[y,x]/(gray_intensity_mean[y,x]*2))

    # 對遮罩區域應用高斯模糊
    # blurred_image = cv2.GaussianBlur(Result_img, (5, 5), 0)

    # # 部分區域高斯模糊
    # blurred_image = np.where(edges[..., np.newaxis] > 0, blurred_image, Result_img)

    # outimg4 = np.hstack([Img1, Result_img])
    #
    # cv2.imshow('Img1_homo', Img1_homo)
    # cv2.imshow('temp1', temp1)
    # cv2.imshow('temp2', temp2)
    # cv2.imshow('temp3', temp3)
    # cv2.waitKey(0)

    return temp1,temp2,temp3

# 合成兩張照片
# def merge_images(Img1,Img2,Img1_bin,homo):
#
#     Img1_bin = cv2.warpPerspective(Img1_bin, homo, (Img1.shape[1], Img1.shape[0]))
#     retval, mask = cv2.threshold(Img1_bin, 1, 255, cv2.THRESH_BINARY_INV)
#     cv2.imshow('Img1_bin', Img1_bin)
#     # cv2.imshow('mask', mask)
#     blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)
#     edges = cv2.Canny(mask, 100, 200)
#     # cv2.imshow('mask', mask)
#     # cv2.imshow('edges', edges)
#     Img1_homo = cv2.warpPerspective(Img1, homo, (Img1.shape[1], Img1.shape[0]))
#     cv2.imshow('Img1_homo', Img1_homo)
#     # cv2.imshow('Img1_bin', Img1_bin)
#     cv2.waitKey(0)
#     h = Img1.shape[0]           # 取得圖片高度
#     w = Img2.shape[1]           # 取得圖片寬度
#
#     for x in range(w):
#         for y in range(h):
#             if (Img1_bin[y,x][0] == 0 and Img1_bin[y,x][1] == 0 and Img1_bin[y,x][2] == 0):
#                 Img1_homo[y,x] = Img2[y,x]
#
#     Result_img = Img1_homo
#     # 對遮罩區域應用高斯模糊
#     blurred_image = cv2.GaussianBlur(Result_img, (5, 5), 0)
#
#     # 部分區域高斯模糊
#     blurred_image = np.where(edges[..., np.newaxis] > 0, blurred_image, Result_img)
#
#     outimg4 = np.hstack([Img1, Result_img])
#     # cv2.imshow("Key Points 2", outimg4)
#     #
#     cv2.imshow('Original Result', Result_img)
#     cv2.imshow('Blurred Result', blurred_image)
#     cv2.waitKey(0)
#
#     return blurred_image

# 計算資料夾內的檔案數量
def file_count(dir_path):
    initial_count = 0
    dir = dir_path
    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            initial_count += 1
    return initial_count

# 自動化處裡檔案內的照片 num可選擇禎數間隔 Feature_model關鍵點檢測方式
def AutoProcessedData(path,num,keypoint_method):
    fileCount = file_count(path)
    # n = int(fileCount / num)
    n = int((fileCount - num) / 10) + 1

    Feature_model = keypoint_select(keypoint_method)
    for i in range(n):
        idx1 = i * 10 + 1
        idx2 = idx1 + num
        if(idx2 < fileCount):
            formatted_idx1 = f'{idx1:03d}'
            formatted_idx2 = f'{idx2:03d}'

            image1_path = path + "/{}.jpg".format(formatted_idx1)
            image2_path = path + "/{}.jpg".format(formatted_idx2)

            print(image1_path)
            print(image2_path)

            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)
            image1 = resize_image(image1)
            image2 = resize_image(image2)
            image1 = sharpen(image1)
            image2 = sharpen(image2)

            kp1, des1, kp2, des2 = Keypoint_detection(image1, image2, Feature_model)
            homo = Feature_matching(image1, image2, kp1, des1, kp2, des2)

            image1, image2 = adjust_brightness(image1, image2)
            # Img1_bin = RemoveReflections(image1)
            merge_image1,merge_image2,merge_image3 = merge_images(image1, image2, homo)
            final_image = np.hstack([image1, image2, merge_image1, merge_image2, merge_image3])
            print(path + '/autoProcess/{}_{}_{}.jpg'.format(keypoint_method,num,i))
            cv2.imwrite(path + '/autoProcess/final_{}_{}_{}.jpg'.format(keypoint_method,num,i), final_image)

# 照片前處理  先縮放成原本大小的一半 在裁切成(540,710)大小
def resize_image(image):
    image_temp = cv2.resize(image, (540, 960), interpolation=cv2.INTER_AREA)
    image_temp = image_temp[250:,:]

    return image_temp

# 選擇關鍵點檢測方法
def keypoint_select(keypoint_method):
    if(keypoint_method == 'orb'):
        orb = cv2.ORB_create()
        Feature_model = orb
    if(keypoint_method == 'brisk'):
        brisk = cv2.BRISK_create()
        Feature_model = brisk
    if(keypoint_method == 'akaze'):
        akaze = cv2.AKAZE_create()
        Feature_model = akaze
    if(keypoint_method == 'sift'):
        sift = cv2.SIFT_create()
        Feature_model = sift
    if(keypoint_method == 'kaze'):
        kaze = cv2.KAZE_create()
        Feature_model = kaze


    return Feature_model


if __name__ == '__main__':
    file_path = "pill_dataset/video_pilldata/output_image/02"

    # 初始化關鍵點模型
    orb = cv2.ORB_create()
    brisk = cv2.BRISK_create()
    akaze = cv2.AKAZE_create()
    sift = cv2.SIFT_create()

    # 關鍵點演算法選擇
    keypoint_method = 'sift'

    AutoProcessedData(file_path,50,keypoint_method)

    # 读取图片found
    # image1 = cv2.imread('pill_dataset/video_pilldata/output_image/02/061.jpg')
    # image2 = cv2.imread('pill_dataset/video_pilldata/output_image/02/111.jpg')
    # image1 = resize_image(image1)
    # image2 = resize_image(image2)
    # image1 = sharpen(image1)
    # image2 = sharpen(image2)
    #
    # # 關鍵點演算法選擇
    # Feature_model = brisk
    # # 尋找關鍵點
    # kp1,des1,kp2,des2 = Keypoint_detection(image1,image2,Feature_model)
    # homo = Feature_matching(image1,image2,kp1,des1,kp2,des2)
    # # -------------------------------------------------
    # # # 二值化後特徵匹配
    # # # image1_Binary = Binary(image1)
    # # # image2_Binary = Binary(image2)
    # # # kp1,des1,kp2,des2 = Keypoint_detection(image1_Binary,image2_Binary,Feature_model)
    # # # homo = Feature_matching(image1_Binary,image2_Binary,kp1,des1,kp2,des2)
    # # --------------------------------------------------
    #
    # image1,image2 = adjust_brightness(image1,image2)
    # # Img1_bin = RemoveReflections(image1)
    # cv2.imshow('image1', image1)
    # cv2.imshow('image2', image2)
    #
    # # cv2.waitKey(0)
    #
    # merge_images(image1, image2, homo)
    # img = np.hstack([image1, image2])
    #
    # # cv2.imshow('drawPoint', img)
    # # cv2.setMouseCallback('drawPoint', show_xy)
    # #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()