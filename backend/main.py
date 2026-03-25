# 모델 불러오기
import io
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor

from lib.utils.path import data_path, model_path
from mybot import ChatHistoryRequest, ChatRequest, chatbot2, mybot

clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # app.state.models = {
    #     'YOLO': create_yolo(),
    #     'SAM3': create_sam3(),
    #     'clip': create_clip(),
    # }
    #
    # print('모델을 불러왔습니다.')

    yield


app = FastAPI(lifespan=lifespan)


def create_yolo():
    return YOLO(model_path('yolo26n.pt'))


def create_sam3():
    overrides = {
        'conf': 0.25,
        'task': 'segment',
        'mode': 'predict',
        'model': model_path('sam3.pt'),
        'half': True,
        'save': True,
    }

    return SAM3SemanticPredictor(overrides=overrides)


def create_clip():
    return (
        CLIPModel.from_pretrained('openai/clip-vit-base-patch32'),
        CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'),
    )


@app.get('/')
async def root():
    return {'message': 'Hello World'}


# 사용자가 이미지를 입력하면 서버는 이미지를 저장한다.
# 파일 저장 이름 바꾸기(오늘날짜-파일이름) hint: datetime
@app.post('/upload_image')
def save_image(file: Annotated[UploadFile, File(...)]):
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
async def predict_yolo(req: Request, file: Annotated[UploadFile, File(...)]):
    # img 읽기
    img = Image.open(io.BytesIO(await file.read()))

    model = req.app.state.models['YOLO']

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


@app.post('/segment_image')
async def segment(
    req: Request,
    file: Annotated[UploadFile, File(...)],
    text_prompt: str = Form('person'),
):
    # 1. 이미지 읽기
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')

    predictor = req.app.state.models['SAM3']
    # 2. 모델 예측 (SAM이나 YOLO-World 등 사용 중인 모델 가정)
    predictor.set_image(np.array(img))
    pred = predictor(text=[text_prompt])[0]

    # 3. 마스크 데이터 추출 (Tensor -> Numpy)
    # pred.masks.data[0] 형태가 [H, W]인 0과 1로 이루어진 마스크라고 가정
    mask_uint8 = (pred.masks.data[0].cpu().numpy() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_uint8)

    # 4. 메모리에 이미지를 PNG로 저장
    img_byte_arr = io.BytesIO()
    mask_img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    # 5. 파일을 직접 반환 (브라우저에서 바로 이미지로 보임)
    return Response(content=img_bytes, media_type='image/png')


classes = ['dog', 'cat', 'pig']


@app.post('/clip_image')
async def clip(file: Annotated[UploadFile, File(...)], text_prompt: str = Form('dog')):
    # 1. 이미지 읽기
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')

    inputs = clip_processor(
        text=classes,
        images=img,
        return_tensors='pt',
        padding=True,
    )

    outputs = clip_model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)

    print(probs[0])

    index = probs[0].argmax()
    print(index)
    name = classes[index]
    print(name)

    return {'index': index.item(), 'classes': classes, 'probs': probs[0].tolist()}


@app.post('/chat')
async def chat(req: ChatRequest):
    print(req.message)
    res = mybot(req.message)

    return {'text': res}


@app.post('/chat_with_history')
async def chat2(req: ChatHistoryRequest):
    response = chatbot2(req.history)

    return {'text': response}
