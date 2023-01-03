import cv2
import csv

pills = []
with open('label.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        pills.append(row[1])
# for k in range(len(pills)):
#     print(str(k)+":"+pills[k])
# 選擇第二隻攝影機
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not (cap.isOpened()):
    print("Could not open video device")
number = 69
count = 31
while True:
    # pill_name = pills[number]
    ret, frame = cap.read()
    pill_name = 'test'

    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32 or k % 256 == 13:
        # SPACE pressed
        img_name = '{}_{}.png'.format(pill_name,count)
        count += 1
        cv2.imwrite("pill_dataset/test_20221228/" + img_name, frame)
        print("{} written!".format(img_name))

    # if(count == 22):
    #     count = 20
    #     number += 1

cap.release()
cv2.destroyAllWindows()
# while True:
#     pill_name = pills[14]
#     ret, frame = cap.read()
#     cv2.imshow(pill_name, frame)
#     k = cv2.waitKey(1)
#     if k % 256 == 32:
#         img_name = f'./dataset/{pill_name}_{img_counter}.png'
#         cv2.imwrite(img_name, frame)
#         print(img_counter)
#         img_counter += 1
#         if img_counter >= 20:
#             # number += 1
#             break
#         # if number >= len(pills):
#         #     break
#
#     if k & 0xFF == ord('q'):
#         break
cap.release()

cv2.destroyAllWindows()
