# 객체 인식 프로그램

OpenCV DNN과 PyQt5를 사용한 실시간 객체 인식 프로그램입니다.

## 📁 프로젝트 구조

```
IPP/
├── simple_detection.py          # 메인 객체 인식 프로그램
├── test_camera.py              # 웹캠 테스트 프로그램
├── download_models.py          # 모델 파일 자동 다운로드
├── project_pre.ipynb           # Jupyter Notebook 버전
├── test.ipynb                  # 테스트용 노트북
├── IPP-2025-1-lab-12-20227123-이선재.ipynb  # 실습 노트북
├── models/                     # AI 모델 파일들 (Git에 포함되지 않음)
│   ├── deploy.prototxt         # 모델 구조 파일
│   └── mobilenet_iter_73000.caffemodel  # 모델 가중치 파일
├── odd/                        # 기타 파일들
├── .gitignore                  # Git 제외 파일 설정
└── README.md                   # 이 파일
```

## 🚀 빠른 시작

### 1. 필요한 패키지 설치
```bash
pip install opencv-python PyQt5 numpy
```

### 2. 모델 파일 다운로드
```bash
python download_models.py
```

### 3. 프로그램 실행
```bash
# 웹캠 테스트 (모델 없이)
python test_camera.py

# 객체 인식 프로그램 (모델 필요)
python simple_detection.py

# Jupyter Notebook 버전
jupyter notebook project_pre.ipynb
```

## 📋 주요 파일 설명

- **`simple_detection.py`**: 실시간 객체 인식 메인 프로그램
- **`test_camera.py`**: 웹캠 연결 테스트용 프로그램
- **`download_models.py`**: AI 모델 파일 자동 다운로드
- **`project_pre.ipynb`**: Jupyter Notebook 버전
- **`models/`**: AI 모델 파일들 (MobileNet SSD)

## 🎯 기능

- 실시간 웹캠 객체 인식
- 20가지 객체 탐지 (사람, 자동차, 고양이, 개, 자전거, 버스 등)
- 조정 가능한 신뢰도 임계값 (50% 이상)
- 사용자 친화적 GUI
- 웹캠 연결 상태 확인

## 🔧 문제 해결

### 모델 파일이 없을 때
```bash
python download_models.py
```

### 웹캠이 안 열릴 때
```bash
# 먼저 웹캠 테스트
python test_camera.py
```

### 수동 모델 다운로드
모델 파일을 수동으로 다운로드하려면:
1. `models/` 폴더 생성
2. [deploy.prototxt](https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt) 다운로드
3. [mobilenet_iter_73000.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel) 다운로드

## 📞 시스템 요구사항

- Python 3.7 이상
- 웹캠 (내장 또는 외장)
- 최소 4GB RAM 권장
- Windows 10/11, macOS, Linux

## 🎓 교육 목적

이 프로젝트는 영상처리 강의의 PyQt 프로젝트로 제작되었습니다.
- OpenCV DNN 모듈 활용
- PyQt5 GUI 개발
- 실시간 비디오 처리
- 객체 인식 알고리즘 구현

## 📝 참고사항

- `models/` 폴더는 Git에 포함되지 않습니다 (용량이 큼)
- 모델 파일들은 자동 다운로드 또는 수동 다운로드 가능
- 웹캠 연결 문제 시 `test_camera.py`로 먼저 확인 