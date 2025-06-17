# AI을 활용한 객체 인식 및 이미지 필터링 프로그램

## 개요
이 프로그램은 OpenCV와 PyQt5를 사용하여 실시간 객체 인식, 이미지 필터링, 그리고 MNIST 숫자 인식 기능을 제공하는 종합적인 영상처리 애플리케이션입니다.

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
pip install opencv-python PyQt5 numpy scikit-learn scipy pandas pillow
```

### 2. 모델 파일 다운로드
```bash
python download_models.py
```

### 3. 프로그램 실행
```bash
python advanced_detection.py
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

## 주요 기능

### 1. 객체 인식 (Object Detection)
- 실시간 웹캠을 통한 객체 인식
- 21가지 클래스 인식 (사람, 자동차, 고양이, 개 등)
- 신뢰도 임계값 조정 가능
- 탐지된 객체 정보 실시간 표시

### 2. 이미지 필터링 (Image Filtering)
- **기본 필터**: 평균, 가우시안, 미디언 필터
- **에지 검출**: 소벨, 라플라시안, 캐니, 프리윗, 로버츠
- **특수 효과**: 웨이브 필터, K-means 클러스터링
- **색상 조정**: 색상 반전, 그레이스케일, 세피아 톤, 색조 조정
- **노이즈 처리**: 가우시안 노이즈 추가

### 3. MNIST 숫자 인식
- **캔버스 기반 숫자 그리기**: 마우스로 숫자를 자유롭게 그리기
- **kNN 기반 예측**: 그린 숫자를 실시간으로 인식
- **별도 데이터 로드**: 시작 시 자동 훈련하지 않고 필요할 때만 로드
- **전처리 자동화**: 그린 이미지를 MNIST 형식으로 자동 변환
- **신뢰도 표시**: 예측 결과와 함께 신뢰도 점수 제공

### 4. 그리기 및 분석 도구
- **그리기 도구**: 선 그리기, 사각형 그리기, 텍스트 추가
- **이미지 분석**: 히스토그램, 색상 통계, 객체 감지, 픽셀 정보
- **이미지 변환**: 회전, 크기 조정

## 파일 구조
```
├── advanced_detection.py    # 메인 프로그램
├── download_models.py       # 객체 인식 모델 다운로드
├── models/                  # 객체 인식 모델 파일들
├── mnist_784.arff          # MNIST 데이터 (이미 포함됨)
└── README.md               # 이 파일
```

## 기술 스택
- **GUI**: PyQt5
- **영상처리**: OpenCV
- **머신러닝**: scikit-learn (kNN)
- **데이터 처리**: NumPy, Pandas
- **이미지 처리**: PIL (Pillow)

## 주의사항
- MNIST 기능을 사용하려면 scikit-learn, scipy, pandas가 필요합니다
- 객체 인식 모델 파일이 models/ 폴더에 있어야 합니다
- MNIST 데이터는 이미 포함되어 있습니다 (mnist_784.arff)

## 라이선스
이 프로젝트는 교육 목적으로 제작되었습니다. 