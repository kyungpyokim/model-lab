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
# # 실습 1. 주민등록증 비식별화 하기 (EasyOCR)

# %%
import easyocr

# %%
# 이슈 해결
## 이슈 1: AttributeError: 'NoneType' object has no attribute 'shape'
### 이미지 경로 영어로 바꾸기
### mac, linux에서는 한글로 해도 됨

## 이슈 2: AttributeError: module 'cv2' has no attribute '__version__'
### 해결 방안: 깃허브로 설치
### uv remove easyocr
### uv add "easyocr @ git+https://github.com/JaidedAI/EasyOCR.git"

# %% [markdown]
# ## 1. 이미지 불러오기

# %%
import matplotlib.pyplot as plt
from PIL import Image

from lib.utils.path import images_path

image_path = str(images_path() / 'id_sample.jpg')
image = Image.open(image_path)

plt.imshow(image)
plt.axis('off')
plt.show()

# %% [markdown]
# ## 2. OCR로 텍스트 추출

# %%
reader = easyocr.Reader(['ko', 'en'])

# %%
results = reader.readtext(image)

results

# %% [markdown]
# ## 이미지에 박스 그리기

# %%
from PIL import ImageDraw

image_boxes = image.copy()
draw = ImageDraw.Draw(image_boxes)

for bbox, text, score in results:
    x1 = int(bbox[0][0])
    y1 = int(bbox[0][1])
    x2 = int(bbox[2][0])
    y2 = int(bbox[2][1])
    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

plt.figure(figsize=(6, 4))
plt.imshow(image_boxes)
plt.axis('off')
plt.title('인식 영역 (빨간 박스)')
plt.show()

# %% [markdown]
# ## 3. 주민등록번호 뒷자리 검은색 박스로 채우기

# %%
results[3]

# %%
image_masked = image.copy()
draw = ImageDraw.Draw(image_masked)

bbox, text, score = results[3]
# 주민번호 패턴 감지: 앞 6자리 - 뒤 7자리
x1 = int(bbox[0][0])
y1 = int(bbox[0][1])
x2 = int(bbox[2][0])
y2 = int(bbox[2][1])

draw.rectangle([x1, y1, x2, y2], fill='black')

plt.figure(figsize=(6, 4))
plt.imshow(image_masked)
plt.axis('off')
plt.title('Result')
plt.show()

# %%
from easyocr.utils import reformat_input

reformat = reformat_input(image)
reformat_path = reformat_input(image_path)

plt.imshow(reformat[0])
plt.axis('off')
plt.show()

# %% [markdown]
# # 실습 2: 실시간 만화 번역하기 (Google Vision Cloud)

# %%
from google.cloud import vision

# %% [markdown]
# ## 서비스 계정 키 등록

# %%
import os

from lib.utils.path import keys_path

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(
    keys_path() / 'google_vision_api_key.json'
)

# %% [markdown]
# ## 클라이언트 불러오기

# %%
client = vision.ImageAnnotatorClient()

# %% [markdown]
# ## 이미지 출력하기

# %%
import matplotlib.pyplot as plt
from PIL import Image

from lib.utils.path import images_path

image_path = images_path() / 'webtoon.jpeg'

image = Image.open(image_path)

plt.imshow(image)
plt.axis('off')
plt.show()

# %% [markdown]
# ## OCR 추출하기

# %%
with open(image_path, 'rb') as f:
    content = f.read()

vision_image = vision.Image(content=content)
res = client.text_detection(image=vision_image)

res

# %%
annotations = res.text_annotations

annotations

# %%
annotations[0]

# %%
annotations[1:]

# %% [markdown]
# ## 이미지에 박스 그리기

# %%
from PIL import ImageDraw

image_boxes = image.copy()
draw = ImageDraw.Draw(image_boxes)

for annot in annotations[1:]:
    v = annot.bounding_poly.vertices
    print(v)
    print('=====')
    print(v[0].x, v[0].y, '|', v[1].x, v[1].y, '|', v[2].x, v[2].y, '|', v[3].x, v[3].y)
    print('=' * 100)

    x1 = v[0].x
    y1 = v[0].y
    x2 = v[2].x
    y2 = v[2].y

    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

# %%
plt.imshow(image_boxes)
plt.axis('off')
plt.show()

# %% [markdown]
# ## 번역하기

# %%
from translate import Translator

translator = Translator(from_lang='ko', to_lang='en')

# \n 으로 나누면 말풍선 덩어리 단위로 분리됨
chunks = [
    line for line in annotations[0].description.strip().split('\n') if line.strip()
]

print('=== 원문 → 번역 ===')
translations = {}
for chunk in chunks:
    translated = translator.translate(chunk)
    translations[chunk] = translated
    print(f'{chunk}  →  {translated}')

# %% [markdown]
# ## 번역 이미지에 나타내기

# %%
# ── 1. annot_dict: 단어 → [x1, y1, x2, y2] ──────────────────────────
annot_dict = {}
for ann in annotations[1:]:
    v = ann.bounding_poly.vertices
    annot_dict[ann.description] = [v[0].x, v[0].y, v[2].x, v[2].y]

annot_dict

# %%
# ── 3. chunk_data: y좌표 기준으로 덩어리 bbox 계산 ───────────────────
import re

Y_TOL = 20

chunk_data = []

for chunk in chunks:
    clean = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', chunk).strip()
    words = clean.split()

    # 청크 단어와 매칭되는 annotation에서 anchor_y 잡기
    anchor_y = None
    for word in words:
        for key, coords in annot_dict.items():
            clean_key = re.sub(r'[^가-힣a-zA-Z0-9]', '', key)
            if clean_key and (word == clean_key or clean_key in word):
                anchor_y = (coords[1] + coords[3]) // 2
                break
        if anchor_y:
            break

    if anchor_y is None:
        continue

    # anchor_y 근처 같은 줄 annotation 모두 묶어 union bbox
    same_line = [
        c for c in annot_dict.values() if abs((c[1] + c[3]) // 2 - anchor_y) <= Y_TOL
    ]

    chunk_data.append(
        {
            'text': chunk,
            'translate': translations.get(chunk, ''),
            'box': [
                min(c[0] for c in same_line),
                min(c[1] for c in same_line),
                max(c[2] for c in same_line),
                max(c[3] for c in same_line),
            ],
        }
    )

chunk_data


# %%
# ── 4. 이미지에 합성 ─────────────────────────────────────────────────
from PIL import ImageFont

FONT_PATH = 'C:/Windows/Fonts/Arial.ttf'
FONT_SIZE = 25
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

image_translated = image.copy()
draw = ImageDraw.Draw(image_translated)

for item in chunk_data:
    x1, y1, x2, y2 = item['box']
    draw.rectangle([x1, y1, x2, y2], fill='white', outline='red')
    draw.text((x1 + 2, y1 + 2), item['translate'], font=font, fill='black')

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].imshow(image)
axes[0].set_title('Original')
axes[0].axis('off')
axes[1].imshow(image_translated)
axes[1].set_title('Translate')
axes[1].axis('off')

plt.tight_layout()
plt.show()

