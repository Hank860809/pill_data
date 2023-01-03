import cv2
import numpy as np
import glob
#
# # path1 測試資料原圖路徑
# # path2 依據我自己lable照片訓練出來的模型辨識測試資料的辨識結果
# # path3 依據教授提供未有反光照片訓練出來的模型辨識測試資料的辨識結果
# # path4 依據教授提供模擬反光照片訓練訓練出來的模型辨識測試資料的辨識結果
# # path1 = './pill_dataset/test/images/test/'
# # path2 = 'runs/detect/exp5/'
# # path3 = 'runs/detect/exp6/'
# # path4 = 'runs/detect/exp7/'
# path1 = 'runs/detect/exp13/'
# path2 = 'runs/detect/exp14/'
# # 調整圖片大小
# img_size = (512,384)
# for i in sorted(glob.glob("./runs/detect/exp14/*.png")):
#     fname = i.split('\\')[1]
#     # img_1 = cv2.imread(i)
#     img_1 = cv2.imread(path1+fname)
#     # cv2.imshow('stack', img_1)
#     # cv2.waitKey(0)
#     img_1 = cv2.resize(img_1, img_size, interpolation=cv2.INTER_AREA)
#     cv2.putText(img_1, "1", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
#     img_2 = cv2.imread(path2+fname)
#     img_2 = cv2.resize(img_2, img_size, interpolation=cv2.INTER_AREA)
#     cv2.putText(img_2, "2", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
#     # img_3 = cv2.imread(path3+fname)
#     # img_3 = cv2.resize(img_3, img_size, interpolation=cv2.INTER_AREA)
#     # cv2.putText(img_3, "3", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
#     # img_4 = cv2.imread(path4+fname)
#     # img_4 = cv2.resize(img_4, img_size, interpolation=cv2.INTER_AREA)
#     # cv2.putText(img_4, "4", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
#     m1 = np.concatenate((img_1, img_2),axis=1)
#     # m2 = np.concatenate((img_3, img_4),axis=1)
#     # m3 = np.concatenate((m1, m2))
#     # cv2.imshow('stack', m3)
#     cv2.imwrite("pill_dataset/Demo/3/" + fname, m1)


path1 = './pill_dataset/dataset'
for i in sorted(glob.glob("./pill_dataset/dataset/2130_*.png")):
    fname = i.split('_')[-1]
    img_1 = cv2.imread(i)
    # cv2.imshow('stack', img_1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("./pill_dataset/dataset/2400_{}".format(fname), img_1)