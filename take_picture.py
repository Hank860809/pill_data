import cv2

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('rtsp://192.168.0.246:8080/h264_pcm.sdp')
cv2.namedWindow("test")

img_counter = 4049_12

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "one_pill{}.png".format(img_counter)
        cv2.imwrite("pill_dataset/test/images/train/" + img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()


# import cv2
#
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
#
# while True:
#     ret, frame = cap.read()
#
#     cv2.imshow("test", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
cv2.destroyAllWindows()