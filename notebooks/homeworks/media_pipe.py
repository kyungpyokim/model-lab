import math

import cv2
import mediapipe as mp

# 모델 불러오기
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 1. 카메라 장치 연결 (0은 기본 내장 캠, 외장 캠은 1 또는 2)
cap = cv2.VideoCapture(0)

# 카메라가 정상적으로 열렸는지 확인
if not cap.isOpened():
    print('카메라를 열 수 없습니다. 연결 상태를 확인하세요.')
    exit()

print("웹캠을 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    # 2. 프레임별로 읽기 (ret: 성공 여부, frame: 이미지 데이터)
    ret, frame = cap.read()

    if not ret:
        print('프레임을 읽어오지 못했습니다.')
        break

    # 3. (선택) 좌우 반전 - 거울처럼 보이게 하기
    flipped_frame = cv2.flip(frame, 1)

    results = hands.process(flipped_frame)

    # 추출 및 그리기
    if results.multi_hand_landmarks:
        ## 손 하나하나 가져오기
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = flipped_frame.shape
            landmarks = hand_landmarks.landmark

            point_1 = None
            point_2 = None

            for idx, landmark in enumerate(landmarks):
                point_x = int(landmark.x * w)
                point_y = int(landmark.y * h)

                if idx == 4:
                    point_1 = (point_x, point_y)
                    cv2.circle(flipped_frame, (point_x, point_y), 5, (0, 255, 0), 2)
                if idx == 8:
                    point_2 = (point_x, point_y)
                    cv2.circle(flipped_frame, (point_x, point_y), 5, (0, 255, 0), 2)

            if point_1 and point_2:
                cv2.line(flipped_frame, point_1, point_2, (255, 0, 0), 2)
                distance = math.dist(point_1, point_2)
                print(f'두 손가락 사이 거리: {distance:.2f}')

    # 4. 화면에 영상 표시
    cv2.imshow('Webcam View', flipped_frame)

    # 5. 'q' 키를 누르면 루프 종료 (1ms 대기)
    # waitKey(1)이 있어야 화면이 실시간으로 갱신됩니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 사용한 자원 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
