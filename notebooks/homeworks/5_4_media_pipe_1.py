## 만약에 웹캠이 켜지지 않는다.
## uv add opencv-contrib-python

## mediapipe 라이브러리 설치 uv add mediapipe==0.10.14
import math
import sys

import cv2
import mediapipe as mp

# 모델 불러오기
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # 근거리, 원거리
    min_detection_confidence=0.5,
)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    refine_landmarks=True,  # 디테일한 포인트 추출
)
##########################################################
# 카메라 시작
##########################################################
vcap = cv2.VideoCapture(0)

# D1675A

while True:
    # 카메라 이미지 추출하기
    ret, frame = vcap.read()

    # 카메라 작동 확인
    if not ret:
        print('카메라가 작동하지 않습니다.')
        sys.exit()

    # 좌우 반전
    flipped_frame = cv2.flip(frame, 1)  # 그릴 도화지 입니다.

    # ------------------------------------------------------
    # 얼굴 그리기 준비
    flipped_frame.flags.writeable = True

    # ------------------------------------------------------
    # Face Detection
    # ------------------------------------------------------
    # # 얼굴 감지하기
    # results = face_detection.process(flipped_frame)

    # # 추출 및 그리기
    # if results.detections:
    #     for detection in results.detections:
    #         mp_drawing.draw_detection(flipped_frame, detection)
    # ------------------------------------------------------
    # Face Mesh
    # ------------------------------------------------------
    # 얼굴 감지하기
    results = face_mesh.process(flipped_frame)

    # 추출 및 그리기
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            point_1 = face_landmarks.landmark[470]
            point_2 = face_landmarks.landmark[472]

            h, w, c = flipped_frame.shape
            p1 = (point_1.x * w, point_1.y * h)
            p2 = (point_2.x * w, point_2.y * h)

            dist_1 = math.dist(p1, p2)
            print(dist_1)
            if dist_1 < 0.1:
                print("Don't Sleep")

            # if i == 470 or i == 472:
            #     print('right eyes')
            #
            # if i == 475 or i == 477:
            #     print('left eyes')
            # mp_drawing.draw_landmarks(
            #     flipped_frame,
            #     face_landmarks,
            #     mp_face_mesh.FACEMESH_TESSELATION,
            #     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            # )

    # 화면 띄우기
    cv2.imshow('webcam', flipped_frame)

    # 화면 끄기
    key = cv2.waitKey(1)
    if key == 27:  # ESC(ASCII Code)
        break

vcap.release()
cv2.destroyAllWindows()
