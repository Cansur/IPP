import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

# 클래스 이름 리스트
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

class SimpleDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("객체 인식 프로그램")
        self.setGeometry(100, 100, 640, 480)

        # UI 구성
        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(320, 240)
        
        self.start_button = QPushButton("시작")
        self.stop_button = QPushButton("정지")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 변수 초기화
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.net = None
        
        # 모델 로딩
        self.load_model()

    def load_model(self):
        """모델을 로드합니다."""
        try:
            print("모델 로딩 중...")
            self.net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", 
                                               "models/mobilenet_iter_73000.caffemodel")
            print("모델 로딩 완료!")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            print("models/ 폴더에 모델 파일이 있는지 확인하세요.")
            self.net = None

    def start_detection(self):
        """객체 인식을 시작합니다."""
        if self.net is None:
            print("모델이 로드되지 않았습니다.")
            return
            
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("웹캠을 열 수 없습니다.")
                return
                
            self.timer.start(50)  # 20FPS
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            print("객체 인식 시작!")
            
        except Exception as e:
            print(f"시작 실패: {e}")

    def stop_detection(self):
        """객체 인식을 정지합니다."""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        print("객체 인식 정지!")

    def update_frame(self):
        """프레임을 업데이트합니다."""
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return

        try:
            # 프레임 크기 조정
            frame = cv2.resize(frame, (320, 240))
            
            # 객체 탐지
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            h, w = frame.shape[:2]

            # 탐지 결과 그리기
            for i in range(min(detections.shape[2], 10)):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    if idx < len(CLASSES):
                        label = CLASSES[idx]
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label}: {confidence:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1)

            # PyQt5 QLabel에 표시
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            
        except Exception as e:
            print(f"프레임 처리 오류: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SimpleDetectionApp()
    win.show()
    sys.exit(app.exec_()) 