import cv2
import numpy as np

# 在圖像上設置一個滑鼠點擊事件的回調函數
def click_event1(event, x, y, flags, param):

    if event == 1:
        points_photo1.append((x, y))
        print("照片一點擊位置：", (x, y))
        # 在點擊的位置畫上紅色的圓點
        cv2.circle(photo1, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Photo 1', photo1)

def click_event2(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        points_photo2.append((x, y))
        print("照片二點擊位置：", (x, y))
        # 在點擊的位置畫上紅色的圓點
        cv2.circle(photo2, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Photo 2', photo2)


def resize_image(image):
    image_temp = cv2.resize(image, (540, 960), interpolation=cv2.INTER_AREA)
    image_temp = image_temp[250:,:]

    return image_temp

def Feature_matching(img1,img2,kp1,des1,kp2,des2,Matcher):
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

    # Matcher = bf_HAMMING

    # 对描述子进行匹配
    # matches = Matcher.match(des1, des2)
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
        print("Not enough matches are found - {}/{}".format(len(good_matches), 10))
        matchesMask = None

    draw_params = dict(singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    # 绘制匹配结果
    # draw_match(img1, img2, kp1, kp2, good_matches, draw_params)

    return homo

# 創建一個空的點集合
points_photo1 = []
points_photo2 = []
#
# 讀取照片
photo1 = cv2.imread('pill_dataset/video_pilldata/output_image/03/061.jpg')
photo1 = resize_image(photo1)
photo2 = cv2.imread('pill_dataset/video_pilldata/output_image/03/111.jpg')
photo2 = resize_image(photo2)
#
# 顯示照片並等待點擊事件
cv2.imshow('Photo 1', photo1)
cv2.setMouseCallback('Photo 1', click_event1)
cv2.imshow('Photo 2', photo2)
cv2.setMouseCallback('Photo 2', click_event2)
cv2.waitKey(0)
# 輸出收集到的點座標
print("照片一 點座標集合：", points_photo1)
print("照片二 點座標集合：", points_photo2)

# 關閉窗口
# cv2.destroyAllWindows()

# 步驟1：標定定位點座標
points_photo1 = np.array(points_photo1)
points_photo2 = np.array(points_photo2)

# 初始化關鍵點模型
orb = cv2.ORB_create()
brisk = cv2.BRISK_create()
akaze = cv2.AKAZE_create()
sift = cv2.SIFT_create()

detectors_name = ['orb','brisk','akaze','sift']
detectors = [orb,brisk,akaze,sift]

# 初始化匹配選擇器 BFMatcher
bf_HAMMING = cv2.BFMatcher(cv2.NORM_HAMMING)
bf_L2 = cv2.BFMatcher(cv2.NORM_L2)

Matcher = bf_HAMMING
# 提取關鍵點和描述符
i = 0
for detector in detectors:
    errors = 0
    average_error = 0
    kp1, des1 = detector.detectAndCompute(photo1, None)  # 可以使用SIFT、ORB、SURF等
    kp2, des2 = detector.detectAndCompute(photo2, None)

    if(i == 3):
        Matcher = bf_L2
    # 步驟2：關鍵點提取和匹配
    try:
        homo = Feature_matching(photo1, photo2, kp1, des1, kp2, des2,Matcher)

        # 步驟3：座標轉換
        new_points = cv2.perspectiveTransform(points_photo1.astype(np.float32).reshape(-1, 1, 2), homo)
        new_points = new_points.reshape(10,2)
        # 步驟4：計算誤差
        errors = np.linalg.norm(new_points - points_photo2, axis=1)
        average_error = np.mean(errors)

        print("{} Average Error:".format(detectors_name[i]), average_error)
    except:
        print('Not enough matches are found')

    i += 1