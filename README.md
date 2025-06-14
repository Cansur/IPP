# 객체 인식 프로젝트

OpenCV DNN을 사용한 객체 인식 프로그램입니다.

## 실행 방법

### 1. 필요한 패키지 설치
```bash
pip install opencv-python PyQt5 numpy
```

### 2. 모델 파일 확인
`models/` 폴더에 다음 파일들이 있어야 합니다:
- `deploy.prototxt`
- `mobilenet_iter_73000.caffemodel`

### 3. 실행
```bash
# Jupyter Notebook으로 실행
jupyter notebook project_pre.ipynb
```

## 주의사항
- `models/` 폴더는 Git에 추가되지 않습니다 (용량이 큼)
- 모델 파일이 없으면 `models/README.md`를 참조하여 다운로드하세요 