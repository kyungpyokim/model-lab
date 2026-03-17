# 모델 불러오기
import io
import shutil
from datetime import datetime

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO

from lib.utils.path import data_path, model_path

model = YOLO(model_path('yolo26n.pt'))
print('모델을 불러왔습니다.')

app = FastAPI()


@app.get('/')
async def root():
    return {'message': 'Hello World'}


# 사용자가 이미지를 입력하면 서버는 이미지를 저장한다.
# 파일 저장 이름 바꾸기(오늘날짜-파일이름) hint: datetime
@app.post('/upload_image')
def save_image(file: UploadFile = File(...)):
    # 파일 이름 설정
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    file_name = f'./images/{now}-{file.filename}'

    # 파일 저장
    with open(file_name, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        'message': '이미지를 저장했습니다.',
        'filename': file_name,
        'time': datetime.now().strftime('%Y%m%d%H%M%S'),
    }


@app.post('/detect_image')
async def predict_yolo(file: UploadFile = File(...)):
    # img 읽기
    img = Image.open(io.BytesIO(await file.read()))

    # 예측하기
    results = model.predict(img)
    result = results[0]

    # 데이터 만들기
    detections = []
    names = result.names
    for x1, y1, x2, y2, conf, label_idx in result.boxes.data:
        detections.append(
            {
                'box': [x1.item(), y1.item(), x2.item(), y2.item()],
                'confidence': conf.item(),
                'label': names[int(label_idx)],
            }
        )

    # 파일 이름 설정
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    file_name = f'{data_path()}/images/{now}-{file.filename}'

    # 파일 저장
    with open(file_name, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        'message': '이미지를 저장했습니다.',
        'filename': file_name,
        'time': datetime.now().strftime('%Y%m%d%H%M%S'),
        'object_detection': detections,
    }
