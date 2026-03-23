import os

import gdown

from lib.utils.path import data_path

url = 'https://drive.google.com/drive/folders/1XwvU5Rfgmg07Gbh44-kkBfZVW-QBNPZF?usp=drive_link'


def download_images(files: list[tuple[str, str]]) -> None:
    # 3. 반복문을 통한 다운로드 실행
    output_dir = str(data_path() / 'images')
    for file_id, file_name in files.items():
        url = f'https://drive.google.com/uc?id={file_id}'
        output_path = os.path.join(output_dir, file_name)

        print(f'\n[시작] {file_name} 다운로드 중...')

        # gdown 실행 (quiet=False로 설정하면 진행 바가 보입니다)
        gdown.download(url, output_path, quiet=False)

    print('\n--- 모든 파일 다운로드가 완료되었습니다. ---')
