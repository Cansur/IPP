#!/usr/bin/env python3
"""
모델 파일 다운로드 스크립트
"""

import os
import urllib.request
import sys

def download_file(url, filename):
    """파일을 다운로드합니다."""
    print(f"{filename} 다운로드 중...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} 다운로드 완료!")
        return True
    except Exception as e:
        print(f"{filename} 다운로드 실패: {e}")
        return False

def setup_models():
    """모델 파일들을 설정합니다."""
    
    # models 폴더 생성
    if not os.path.exists("models"):
        os.makedirs("models")
        print("models 폴더 생성 완료")
    
    # 모델 파일들
    model_files = {
        "models/deploy.prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
        "models/mobilenet_iter_73000.caffemodel": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
    }
    
    # 파일이 없으면 다운로드
    for filename, url in model_files.items():
        if not os.path.exists(filename):
            if not download_file(url, filename):
                print(f"오류: {filename} 다운로드에 실패했습니다.")
                return False
        else:
            print(f"{filename} 이미 존재합니다.")
    
    print("\n모든 모델 파일이 준비되었습니다!")
    print("이제 'python simple_detection.py'로 실행할 수 있습니다.")
    return True

if __name__ == "__main__":
    print("=== 모델 파일 다운로드 ===")
    setup_models() 