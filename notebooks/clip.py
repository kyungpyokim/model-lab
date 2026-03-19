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
# # 실습 1: 이미지 분류

# %% [markdown]
# ## 1) Ultralytics 버전

# %%
# 설치 uv pip install git+https://

# %% [markdown]
# ### 모델 불러오기

# %%
import clip
import torch
from PIL import Image

from lib.utils.device import available_device

device = available_device()

model, processes = clip.load('ViT-B/32', device=device)

# %% [markdown]
# ### 이미지/텍스트 준비하기

# %%
from lib.utils.path import images_path

image_path = images_path() / 'badugi.jpg'
classes = ['a person', 'a car', 'a dog', 'a cat']

img = Image.open(image_path)

# %% [markdown]
# ### 모델에 넣을 준비

# %%
image = processes(img).unsqueeze(0).to(device)
text = clip.tokenize(classes).to(device)

# %% [markdown]
# ### 유사도 계산하기

# %%
with torch.no_grad():
    logits_per_image, _ = model(image, text)
    print(logits_per_image)
    print('=' * 20)
    print(_)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    print('=' * 20)
    print(probs)

# %% [markdown]
# ### 결과

# %%
for label, prob in zip(classes, probs, strict=False):
    print(f'{label}: {prob:.4f}')

# %%
probs_test = torch.tensor([1.325e-02, 6.704e-04, 9.736e-01, 1.226e-02])
classes = ['a person', 'a car', 'a dog', 'a cat']

print(probs_test.argmax())
print(classes[probs_test.argmax()])


# %%
print(1.325e-02, 9.736e-01)

# %% [markdown]
# #### unsqueeze란?

# %%
import torch

x1 = torch.tensor([1.0, 2.0, 3.0])
print(x1)
print(x1.shape)

x2 = x1.unsqueeze(0)
print(x2)
print(x2.shape)

# %% [markdown]
# ## 2) HuggingFace 버전

# %% [markdown]
# ### 모델 불러오기

# %%
import requests
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')


# %% [markdown]
# ### 이미지/텍스트 준비하기

# %%
from PIL import Image

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
classes = ['a photo of a cat', 'a photo of a dog']


# %% [markdown]
# ### 모델에 넣을 준비

# %%
inputs = processor(
    text=classes,
    images=image,
    return_tensors='pt',
    padding=True,
)


# %% [markdown]
# ### 유사도 계산

# %%
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(
    dim=1
)  # we can take the softmax to get the label probabilities


# %%
print(probs)

# %%
print(probs.squeeze(0))

# %%
for label, prob in zip(classes, probs[0], strict=False):
    print(f'{label}: {prob:.4f}')

# %%
index = probs[0].argmax()
print(index)
name = classes[index]
print(name)

# %%
import matplotlib.pyplot as plt

plt.imshow(image)
plt.title(name)
plt.axis('off')
plt.show()

# %% [markdown]
# # 실습 2: 이미지 검색

# %% [markdown]
# ### 모델 불러오기

# %%
from transformers import CLIPModel, CLIPProcessor

from lib.utils.device import available_device

device = available_device()

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# %% [markdown]
# ### 검색 대상 폴더 불어오기

# %%
import os

from lib.utils.path import images_path

image_folder = images_path() / 'clothes'
files = []

allow_extensions = ['jpg', 'jpeg', 'png']
for file in os.listdir(image_folder):
    if file.split('.')[-1] in allow_extensions:
        files.append(os.path.join(image_folder, file))

print(files)


# %% [markdown]
# ### 폴더 이미지 벡터화 하기

# %%
import torch
from PIL import Image

image_features = []

for file in files:
    image = Image.open(file)
    inputs = processor(images=image, return_tensors='pt').to(device)

    with torch.no_grad():
        feature = model.vision_model(pixel_values=inputs['pixel_values']).pooler_output
        feature = model.visual_projection(feature)
        feature /= feature.norm(dim=-1, keepdim=True)

    image_features.append(feature.cpu())

image_features = torch.cat(image_features, dim=0)

# %%
# # cat 예시
feat1 = torch.tensor([[1.0, 2.0, 3.0]])
feat2 = torch.tensor([[4.0, 5.0, 6.0]])

features = [feat1, feat2]
print(features)

features_cat = torch.cat(features, dim=0)
print(features_cat)

# %% [markdown]
# ### 검색하기

# %%
query = 'a blue shirt'
query = 'a black shirt'
top_k = 3

# %%
# 텍스트 벡터화
inputs = processor(text=[query], return_tensors='pt').to(device)

with torch.no_grad():
    text_feature = model.text_model(
        input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']
    ).pooler_output

    text_feature = model.text_projection(text_feature)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)

# %%
# 유사도 계산
similarity = (image_features @ text_feature.cpu().T).squeeze(1)  # cos 유사도 공식
top_indices = similarity.argsort(descending=True)[:top_k]  # 정렬 순서를 위치로 반환

print(similarity)
print(files)
print(top_indices)

# %%
import matplotlib.pyplot as plt

img = Image.open(files[2])

plt.imshow(img)
plt.title(query)
plt.axis('off')
plt.show()

# %% [markdown]
# # 실습 3: 모델 유합(YOLO + CLIP)

# %%
# 나는 이미지에서 사람이 앉아있는지 아닌지를 분류하고 싶습니다.
# sit.jpg
# YOLO로 사람 찾기
# CLIP 1을 활용해서 사람 분류하기

# %% [markdown]
# ### 모델 불러오기

# %%
from ultralytics import YOLO

from lib.utils.path import model_path

yolo_model = YOLO(model_path('yolo26n.pt'))

# %%
from transformers import CLIPModel, CLIPProcessor

clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# %%
from lib.utils.device import available_device

device = available_device()

yolo_model.to(device)
clip_model.to(device)

# %% [markdown]
# ### 이미지/텍스트 준비하기

# %%
from PIL import Image

from lib.utils.path import images_path

image_path = images_path() / 'sit.jpg'
classes = ['a person standing', 'a person sitting']

# %% [markdown]
# ### YOLO로 객체 감지하기

# %%
yolo_pred = yolo_model(source=image_path)[0]
yolo_pred

# %%
names = yolo_pred.names
boxes = yolo_pred.boxes

persons = []

for data in boxes.data:
    x1, y1, x2, y2, conf, idx = data.cpu()
    if names[int(idx)] == 'person':
        persons.append(data.cpu())

# %% [markdown]
# ### 객체 감지한 결과에서 CLIP으로 분류하기

# %%
import matplotlib.pyplot as plt

x1, y1, x2, y2, conf, idx = persons[0]
crop_image = image.crop([int(x1), int(y1), int(x2), int(y2)])

plt.imshow(crop_image)
plt.axis('off')
plt.show()

# %%
from PIL import Image

image = Image.open(image_path)

for data in persons:
    x1, y1, x2, y2, conf, idx = data
    crop_image = image.crop([int(x1), int(y1), int(x2), int(y2)])

    inputs = processor(
        text=classes, images=crop_image, return_tensors='pt', padding=True
    ).to(device)

    with torch.no_grad():
        output = clip_model(**inputs)
        logits_per_image = output.logits_per_image
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    best = classes[probs.argmax()]
    print(f'{best} {probs.max(): .4f}')
