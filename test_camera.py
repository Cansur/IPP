import cv2
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class CameraTestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("웹캠 테스트")
        self.setGeometry(100, 100, 640, 480)

        # UI 구성
        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(320, 240)
        
        self.start_button = QPushButton("웹캠 시작")
        self.stop_button = QPushButton("웹캠 정지")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

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

    def start_camera(self):
        """웹캠을 시작합니다."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("웹캠을 열 수 없습니다.")
                return
                
            self.timer.start(50)  # 20FPS
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            print("웹캠 시작!")
            
        except Exception as e:
            print(f"웹캠 시작 실패: {e}")

    def stop_camera(self):
        """웹캠을 정지합니다."""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        print("웹캠 정지!")

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
    win = CameraTestApp()
    win.show()
    sys.exit(app.exec_()) 