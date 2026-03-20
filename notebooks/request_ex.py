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

# %%
import requests

url = 'http://localhost:8000'

res = requests.get(url)
print(res.status_code)

# %%
from lib.utils.path import data_path

with open(data_path() / 'images' / 'bus.jpg', 'rb') as f:
    files = {'file': f}

    res = requests.post(url=f'{url}/upload_image', files=files)

    if res.status_code == 200:
        print(res.json())

# %%
import json

from lib.utils.path import data_path

with open(data_path() / 'images' / 'bus.jpg', 'rb') as f:
    files = {'file': f}

    res = requests.post(url=f'{url}/detect_image', files=files)

    if res.status_code == 200:
        print(json.dumps(res.json(), indent=4))

# %% [markdown]
# # 3. detect_image 테스트

# %%

# %%
import json

import requests

from lib.utils.path import data_path

url = 'http://localhost:8000'

with open(data_path() / 'images' / 'badugi.jpg', 'rb') as f:
    files = {'file': f}

    res = requests.post(url=f'{url}/clip_image', files=files)

    if res.status_code == 200:
        print(json.dumps(res.json(), indent=4))
