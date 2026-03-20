import sys

import cv2
import matplotlib.pyplot as plt

vcap = cv2.VideoCapture(0)

while True:
    ret, frame = vcap.read()

    if not ret:
        print('카메라가 작동하지 않습니다.')
        sys.exit()

    flipped_frame = cv2.flip(frame, 1)

    # cv2.imshow('webcam', filpped_frame)
    plt.imshow(cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB))
    plt.show()

    # cv2.imshow('webcam', flipped_frame)

    # 중요: 'q' 키를 누르면 루프 탈출 (1ms 대기)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vcap.release()
cv2.destroyAllWindows()
