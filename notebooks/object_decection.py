# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: model_lab (3.11.14)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1. 모델 불러오기

# %%
# 라이브러리 설치 : Uv add ultralytics
from ultralytics import YOLO

model = YOLO('../data/models/yolo26n.pt')
model


# %% [markdown]
# # 2. 이미지 예측하기

# %%
from lib.utils.path import data_path

preds = model(source=data_path() / 'images/yolo_image.jpg', save=True)

preds

# %%
print(len(preds))
print(type(preds))

# %%
pred = preds[0]
print(pred.names)


# %%
print(pred.boxes)

# %%
print(f'data: {pred.boxes.data[0]}')


# %%
names = pred.names

x1, y1, x2, y2, conf, predict = pred.boxes.data[0]

# %%
print(pred.boxes)

# %%
from PIL import Image, ImageDraw, ImageFont

img = Image.open('./images/yolo_image.jpg')
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('C:/Windows/Fonts/ARIAL.TTF', size=50)
r = 10  # 반지름

x1, y1, x2, y2, conf, predict = pred.boxes.data[0]
# 왼쪽 위 좌표 점으로 표시하기
draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill='red')
# 오른쪽 좌표 점으로 표시하기
draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill='blue')
# 박스 그리기
draw.rectangle([x1, y1, x2, y2], outline='black', width=3)
# 라벨 출력
draw.text((x1, y1), text=names[int(predict)], fill='black', font=font)

img

# %%
names = pred.names

for data in pred.boxes.data:
    x1, y1, x2, y2, conf, predict = data
    print(f'왼쪽 위 좌표: ({x1}, {y1})')
    print(f'오른쪽 아래 좌표: ({x2}, {y2})')
    print(f'Confidence: {conf}')
    print(f'객체 분류 결과: {predict} | {names[int(predict)]}')

# %%
# index3번째 객체의 cls, conf, data 출력하기 + cls 가 뭘 의미하는지 names에서 찾기

names = pred.names

print('3번째 객체 정보')
print(f'cls: {pred.boxes.cls[3]} name : {names[int(pred.boxes.cls[3])]}')
print(f'conf: {pred.boxes.conf[3]}')
print(f'data: {pred.boxes.data[3]}')


# %% [markdown]
# # 3. 모델 학습하기

# %% [markdown]
# ## 1) Roboflow에서 데이터 불러오기

# %%
from roboflow import Roboflow

rf = Roboflow(api_key='rEPkMkyyUp3cshcgdhgv')
project = rf.workspace('kyungpyos-workspace').project('dogcatproject-tupje')
version = project.version(1)
dataset = version.download('yolo26')

# %% [markdown]
# ## 2) 모델 불러오기

# %%
from ultralytics import YOLO

from lib.utils.path import model_path

model = YOLO(model_path('yolo26n.pt'))

# %% [markdown]
# ## 3) 학습하기

# %%
results = model.train(
    data='./DogCatProject-1/data.yaml',  # data.yaml 파일 경로
    epochs=100,  #
    imgsz=640,
    project='yolo_run',
)

# %% [markdown]
# # 4. 내 모델 불러오기

# %%
from ultralytics import YOLO

model = YOLO('../runs/detect/yolo_run/train2/weights/best.pt')

# %% [markdown]
# # 5. 예측 테스트

# %%
from lib.utils.path import data_path

results = model.predict(source=data_path() / 'images' / 'badugi.jpg', save=True)
