import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def load_image(filename: str, color_channel : str ="RGB") -> np.ndarray:
    """파일에서 이미지를 로드합니다.

    Args:
        filename (str): 이미지 파일 경로
    Returns:
        np.ndarray: 로드된 이미지 데이터
    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        ValueError: 이미지를 로드할 수 없을 경우
    """
    if not os.path.exists(filename):
        # raise는 함수안에서 사용을 한다고 하여도 main body에서 작용되는 글로벌로 작용된다?
        raise FileNotFoundError(f"File {filename} does not exist")
    
    # 삼항 연산자
    # imread = plt.imread if color_channel == "RGB" else cv2.imread
    
    if color_channel == "RGB":
        # 함수 자체를 넣을 수 있음
        imread = plt.imread
    elif color_channel == "BGR":
        imread = cv2.imread
    else:
        raise ValueError(f"undifined color_channel {color_channel}")
    img: np.ndarray = cv2.imread(filename)
    if img is None:
        return ValueError(f"Could Not load {filename}")
    return img
