import cv2
import os
import numpy as np

def load_image(filename: str, color_channel: str = "RGB") -> np.ndarray:
    """
    파일에서 이미지를 로드합니다.

    Args:
        filename (str): 이미지 파일 경로
        color_channel (str): 색상 채널 형식 ("RGB" 또는 "BGR")
    Returns:
        np.ndarray: 로드된 이미지 데이터 (dtype=uint8, 3채널)
    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        ValueError: 이미지를 로드할 수 없거나 잘못된 color_channel이 지정된 경우
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"파일 {filename}이 존재하지 않습니다")

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"{filename}을 로드할 수 없습니다")

    if color_channel == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_channel != "BGR":
        raise ValueError(f"지원하지 않는 색상 채널: {color_channel}")

    # dtype과 연속성 보장
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    img = np.ascontiguousarray(img)

    return img
