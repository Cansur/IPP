import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QSlider, QMenuBar, QMenu, QAction, QFileDialog, QInputDialog,
    QTabWidget, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QFont, QColor

# 클래스 이름 리스트
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

class AdvancedDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("고급 객체 인식 및 이미지 필터링 프로그램")
        self.setGeometry(100, 100, 1600, 900)  # 창 크기 증가

        # 변수 초기화
        self.cap = None
        self.net = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.filtered_frame = None
        self.current_filter = "None"
        self.click_point = None
        self.drawing = False
        self.last_point = None
        self.draw_mode = None
        self.angle = 0
        self.confidence_threshold = 0.5

        # UI 설정
        self.setup_ui()
        self.create_menu_bar()
        
        # 모델 로딩
        self.load_model()

    def setup_ui(self):
        # 메인 위젯과 레이아웃
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # 탭 위젯 생성
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        # 객체 인식 탭
        self.detection_tab = QWidget()
        self.detection_layout = QHBoxLayout(self.detection_tab)
        
        # 왼쪽 패널 (이미지 표시)
        left_panel = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        left_panel.addWidget(self.image_label)

        # 컨트롤 버튼
        self.control_layout = QHBoxLayout()
        self.start_button = QPushButton("시작")
        self.stop_button = QPushButton("정지")
        self.stop_button.setEnabled(False)
        
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        
        self.control_layout.addWidget(self.start_button)
        self.control_layout.addWidget(self.stop_button)
        left_panel.addLayout(self.control_layout)

        # 오른쪽 패널 (설정)
        right_panel = QVBoxLayout()
        
        # 신뢰도 임계값 설정
        confidence_group = QGroupBox("객체 인식 설정")
        confidence_layout = QVBoxLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(1)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        confidence_layout.addWidget(QLabel("신뢰도 임계값"))
        confidence_layout.addWidget(self.confidence_slider)
        confidence_group.setLayout(confidence_layout)
        right_panel.addWidget(confidence_group)

        # 탐지된 객체 정보
        self.detection_info = QLabel("탐지된 객체 정보가 여기에 표시됩니다.")
        self.detection_info.setWordWrap(True)
        right_panel.addWidget(self.detection_info)

        right_panel.addStretch()
        
        # 레이아웃 결합
        self.detection_layout.addLayout(left_panel, 2)
        self.detection_layout.addLayout(right_panel, 1)

        # 이미지 필터링 탭
        self.filter_tab = QWidget()
        self.filter_layout = QHBoxLayout(self.filter_tab)
        
        # 왼쪽 패널 (이미지 표시)
        filter_left = QVBoxLayout()
        self.filtered_label = QLabel()
        self.filtered_label.setMinimumSize(640, 480)
        self.filtered_label.mousePressEvent = self.set_click_point
        self.filtered_label.mouseMoveEvent = self.draw_on_image
        self.filtered_label.mouseReleaseEvent = self.stop_drawing
        filter_left.addWidget(self.filtered_label)

        # 필터 크기 슬라이더
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(51)
        self.slider.setValue(3)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(5)
        self.slider.valueChanged.connect(self.update_filtered_image)
        filter_left.addWidget(QLabel("필터 크기"))
        filter_left.addWidget(self.slider)

        # 회전 각도 슬라이더
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setMinimum(-180)
        self.angle_slider.setMaximum(180)
        self.angle_slider.setValue(0)
        self.angle_slider.setTickPosition(QSlider.TicksBelow)
        self.angle_slider.setTickInterval(15)
        self.angle_slider.valueChanged.connect(self.rotate_image)
        filter_left.addWidget(QLabel("회전 각도"))
        filter_left.addWidget(self.angle_slider)

        # 오른쪽 패널 (필터 설정)
        filter_right = QVBoxLayout()
        
        # 필터 그룹
        filter_group = QGroupBox("필터 설정")
        filter_settings = QGridLayout()
        
        # 노이즈 설정
        self.noise_amount = QDoubleSpinBox()
        self.noise_amount.setRange(0, 100)
        self.noise_amount.setValue(25)
        self.noise_amount.setSingleStep(5)
        filter_settings.addWidget(QLabel("노이즈 강도:"), 0, 0)
        filter_settings.addWidget(self.noise_amount, 0, 1)
        
        # K-means 클러스터 설정
        self.k_clusters = QSpinBox()
        self.k_clusters.setRange(2, 20)
        self.k_clusters.setValue(8)
        self.k_clusters.setSingleStep(1)
        filter_settings.addWidget(QLabel("클러스터 수 (K):"), 1, 0)
        filter_settings.addWidget(self.k_clusters, 1, 1)
        self.k_clusters.valueChanged.connect(self.update_filtered_image)
        
        # 웨이브 필터 설정
        self.wave_freq = QDoubleSpinBox()
        self.wave_freq.setRange(0.1, 2.0)
        self.wave_freq.setValue(0.5)
        self.wave_freq.setSingleStep(0.1)
        filter_settings.addWidget(QLabel("파동 주파수:"), 2, 0)
        filter_settings.addWidget(self.wave_freq, 2, 1)
        
        self.wave_amp = QSpinBox()
        self.wave_amp.setRange(1, 20)
        self.wave_amp.setValue(5)
        filter_settings.addWidget(QLabel("파동 진폭:"), 3, 0)
        filter_settings.addWidget(self.wave_amp, 3, 1)
        
        filter_group.setLayout(filter_settings)
        filter_right.addWidget(filter_group)
        
        # 분석 그룹
        analysis_group = QGroupBox("이미지 분석")
        analysis_layout = QVBoxLayout()
        self.analysis_info = QLabel("이미지 분석 정보가 여기에 표시됩니다.")
        self.analysis_info.setWordWrap(True)
        analysis_layout.addWidget(self.analysis_info)
        analysis_group.setLayout(analysis_layout)
        filter_right.addWidget(analysis_group)
        
        filter_right.addStretch()
        
        # 레이아웃 결합
        self.filter_layout.addLayout(filter_left, 2)
        self.filter_layout.addLayout(filter_right, 1)

        # 탭 추가
        self.tab_widget.addTab(self.detection_tab, "객체 인식")
        self.tab_widget.addTab(self.filter_tab, "이미지 필터링")

    def create_menu_bar(self):
        menubar = self.menuBar()

        # 파일 메뉴
        file_menu = menubar.addMenu("파일")
        open_action = QAction("이미지 열기", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        save_action = QAction("이미지 저장", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        # 필터 메뉴
        filter_menu = menubar.addMenu("필터")
        filters = [
            ("평균 필터", "Mean", "이미지를 부드럽게 합니다."),
            ("가우시안 블러", "Gaussian", "가우시안 분포로 부드럽게 합니다."),
            ("미디언 필터", "Median", "노이즈를 효과적으로 제거합니다."),
            ("소벨 필터", "Sobel", "에지를 강조합니다."),
            ("라플라시안 필터", "Laplacian", "에지를 선명하게 합니다."),
            ("웨이브 필터", "Wave", "파동 효과를 적용합니다."),
            ("K-means 클러스터링", "KMeans", "이미지의 색상을 K개의 클러스터로 양자화합니다.")
        ]
        for name, mode, tooltip in filters:
            action = QAction(name, self)
            action.setToolTip(tooltip)
            action.setData(mode)
            action.triggered.connect(self.set_filter)
            filter_menu.addAction(action)

        # 에지 검출 메뉴
        edge_menu = menubar.addMenu("에지 검출")
        edges = [
            ("캐니 엣지", "Canny", "캐니 알고리즘으로 에지를 검출합니다."),
            ("프리윗 엣지", "Prewitt", "프리윗 필터로 에지를 검출합니다."),
            ("로버츠 엣지", "Roberts", "로버츠 필터로 에지를 검출합니다.")
        ]
        for name, mode, tooltip in edges:
            action = QAction(name, self)
            action.setToolTip(tooltip)
            action.setData(mode)
            action.triggered.connect(self.apply_edge_filter)
            edge_menu.addAction(action)

        # 노이즈 메뉴
        noise_menu = menubar.addMenu("노이즈")
        noise_action = QAction("가우시안 노이즈 추가", self)
        noise_action.setToolTip("이미지에 가우시안 노이즈를 추가합니다.")
        noise_action.triggered.connect(self.add_gaussian_noise)
        noise_menu.addAction(noise_action)

        # 색상 메뉴
        color_menu = menubar.addMenu("색상")
        color_actions = [
            ("색상 반전", self.invert_colors, "이미지의 색상을 반전시킵니다."),
            ("그레이스케일", self.to_grayscale, "이미지를 흑백으로 변환합니다."),
            ("세피아 톤", self.apply_sepia, "이미지에 세피아 톤을 적용합니다."),
            ("색조 조정", self.adjust_hue, "이미지의 색조를 조정합니다.")
        ]
        for name, func, tooltip in color_actions:
            action = QAction(name, self)
            action.setToolTip(tooltip)
            action.triggered.connect(func)
            color_menu.addAction(action)

        # 그리기 메뉴
        draw_menu = menubar.addMenu("그리기")
        draw_actions = [
            ("선 그리기", self.enable_line_drawing, "마우스로 자유롭게 선을 그립니다."),
            ("사각형 그리기", self.draw_rectangle, "중심점을 기준으로 사각형을 그립니다."),
            ("텍스트 추가", self.add_text, "이미지에 텍스트를 추가합니다."),
            ("랜덤 도형 효과", self.random_shapes, "무작위 도형을 추가하여 예술적 효과를 만듭니다.")
        ]
        for name, func, tooltip in draw_actions:
            action = QAction(name, self)
            action.setToolTip(tooltip)
            action.triggered.connect(func)
            draw_menu.addAction(action)

        # 분석 메뉴
        analysis_menu = menubar.addMenu("분석")
        analysis_actions = [
            ("히스토그램 표시", self.show_histogram, "이미지의 밝기 분포를 분석합니다."),
            ("색상 통계", self.color_statistics, "이미지의 평균 및 주요 색상을 계산합니다."),
            ("객체 감지", self.detect_objects, "주요 윤곽선을 감지하여 표시합니다."),
            ("픽셀 정보", self.show_pixel_info, "클릭한 지점의 픽셀 색상 정보를 표시합니다.")
        ]
        for name, func, tooltip in analysis_actions:
            action = QAction(name, self)
            action.setToolTip(tooltip)
            action.triggered.connect(func)
            analysis_menu.addAction(action)

    def load_model(self):
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
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        print("객체 인식 정지!")

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return

        try:
            # 프레임 크기 조정
            frame = cv2.resize(frame, (640, 480))
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 객체 탐지
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            h, w = frame.shape[:2]
            detected_objects = []

            # 탐지 결과 그리기
            for i in range(min(detections.shape[2], 10)):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    idx = int(detections[0, 0, i, 1])
                    if idx < len(CLASSES):
                        label = CLASSES[idx]
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label}: {confidence:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1)
                        detected_objects.append(f"{label}: {confidence:.2f}")

            # 탐지된 객체 정보 업데이트
            if detected_objects:
                self.detection_info.setText("탐지된 객체:\n" + "\n".join(detected_objects))
            else:
                self.detection_info.setText("탐지된 객체가 없습니다.")

            # PyQt5 QLabel에 표시
            self.display_image(frame, self.image_label)
            
        except Exception as e:
            print(f"프레임 처리 오류: {e}")

    def open_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "이미지 열기", "", 
                                                "이미지 파일 (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            # BGR로 이미지 로드
            img_bgr = cv2.imread(filename)
            if img_bgr is None:
                print("이미지를 불러올 수 없습니다.")
                return
            # BGR에서 RGB로 변환
            self.current_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.filtered_frame = self.current_frame.copy()
            self.update_filtered_image()
            self.display_image(self.filtered_frame, self.filtered_label)

    def set_filter(self):
        self.current_filter = self.sender().data()
        self.update_filtered_image()

    def apply_wave_filter(self, image):
        """웨이브 필터를 적용합니다."""
        img_pil = Image.fromarray(image).convert('L').resize((800, 800))
        width, height = img_pil.size
        pixels = np.array(img_pil)
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        pen = QPen(Qt.black, 1)
        painter.setPen(pen)
        frequency = self.wave_freq.value()
        max_amplitude = self.wave_amp.value()
        num_lines = 100
        line_spacing = height // num_lines
        all_points = []
        for line_idx in range(num_lines):
            y_base = line_idx * line_spacing
            points = []
            for x in range(width):
                y_pixel = min(max(y_base, 0), height - 1)
                brightness = pixels[y_pixel, x] / 255.0
                amplitude = (1 - brightness) * max_amplitude
                wave_offset = amplitude * np.sin(frequency * x)
                y_pos = y_base + wave_offset
                points.append(QPoint(x, int(y_pos)))
            all_points.append(points)
        for i in range(len(all_points) - 1):
            current_points = all_points[i]
            next_points = all_points[i + 1]
            painter.drawLine(current_points[-1], next_points[-1]) if i % 2 == 0 else painter.drawLine(current_points[0], next_points[0])
            for j in range(len(current_points) - 1):
                painter.drawLine(current_points[j], current_points[j + 1])
        painter.end()
        qimage = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        width, height = qimage.width(), qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr.asstring(), np.uint8).reshape((height, width, 3))
        return arr

    def kmeans_color_quantization(self, img, K):
        """K-means 클러스터링을 사용하여 이미지의 색상을 양자화합니다."""
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts = 10

        ret, labels, centers = cv2.kmeans(Z, K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized_img = quantized.reshape((img.shape))

        return quantized_img

    def update_filtered_image(self):
        if self.current_frame is None:
            return
            
        kernel_size = self.slider.value()
        if self.current_filter not in ["Sobel", "Laplacian", "Wave", "KMeans"]:
            kernel_size = kernel_size - 1 if kernel_size % 2 == 0 else kernel_size
            kernel_size = max(1, kernel_size)

        if self.current_filter == "KMeans":
            # K-means의 경우 RGB 형식으로 직접 처리
            k = self.k_clusters.value()
            self.filtered_frame = self.kmeans_color_quantization(self.current_frame, k)
        else:
            # 다른 필터들은 BGR 형식으로 처리
            img_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            
            if self.current_filter == "Mean":
                self.filtered_frame = cv2.blur(img_bgr, (kernel_size, kernel_size))
            elif self.current_filter == "Gaussian":
                self.filtered_frame = cv2.GaussianBlur(img_bgr, (kernel_size, kernel_size), 0)
            elif self.current_filter == "Median":
                self.filtered_frame = cv2.medianBlur(img_bgr, kernel_size)
            elif self.current_filter == "Sobel":
                sobel_x = cv2.Sobel(img_bgr, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(img_bgr, cv2.CV_64F, 0, 1, ksize=3)
                sobel_x = cv2.convertScaleAbs(sobel_x)
                sobel_y = cv2.convertScaleAbs(sobel_y)
                self.filtered_frame = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
            elif self.current_filter == "Laplacian":
                self.filtered_frame = cv2.Laplacian(img_bgr, cv2.CV_64F, ksize=3)
                self.filtered_frame = cv2.convertScaleAbs(self.filtered_frame)
            elif self.current_filter == "Wave":
                self.filtered_frame = self.apply_wave_filter(self.current_frame)
            
            if self.current_filter != "Wave":
                self.filtered_frame = cv2.cvtColor(self.filtered_frame, cv2.COLOR_BGR2RGB)
        
        self.display_image(self.filtered_frame, self.filtered_label)

    def invert_colors(self):
        if self.current_frame is None:
            return
        self.filtered_frame = cv2.bitwise_not(self.current_frame)
        self.display_image(self.filtered_frame, self.filtered_label)

    def to_grayscale(self):
        if self.current_frame is None:
            return
        self.filtered_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY)
        self.filtered_frame = cv2.cvtColor(self.filtered_frame, cv2.COLOR_GRAY2RGB)
        self.display_image(self.filtered_frame, self.filtered_label)

    def apply_sepia(self):
        if self.current_frame is None:
            return
        img_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
        sepia_matrix = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        self.filtered_frame = cv2.transform(img_bgr, sepia_matrix)
        self.filtered_frame = np.clip(self.filtered_frame, 0, 255).astype(np.uint8)
        self.filtered_frame = cv2.cvtColor(self.filtered_frame, cv2.COLOR_BGR2RGB)
        self.display_image(self.filtered_frame, self.filtered_label)

    def display_image(self, image, label):
        if image is None:
            return
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.display_image(self.current_frame, self.image_label)
        if self.filtered_frame is not None:
            self.display_image(self.filtered_frame, self.filtered_label)

    def update_confidence(self):
        self.confidence_threshold = self.confidence_slider.value() / 100.0

    def save_image(self):
        if self.filtered_frame is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "이미지 저장", "",
                                                "PNG 이미지 (*.png);;JPEG 이미지 (*.jpg);;BMP 이미지 (*.bmp)")
        if filename:
            cv2.imwrite(filename, cv2.cvtColor(self.filtered_frame, cv2.COLOR_RGB2BGR))

    def apply_edge_filter(self):
        if self.current_frame is None:
            return
        mode = self.sender().data()
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY)
        if mode == "Canny":
            edges = cv2.Canny(gray, 100, 200)
        elif mode == "Prewitt":
            kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32)
            kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)
            edges = cv2.filter2D(gray, -1, kernelx) + cv2.filter2D(gray, -1, kernely)
        elif mode == "Roberts":
            kernelx = np.array([[1,0],[0,-1]], dtype=np.float32)
            kernely = np.array([[0,1],[-1,0]], dtype=np.float32)
            edges = cv2.filter2D(gray, -1, kernelx) + cv2.filter2D(gray, -1, kernely)
        self.filtered_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        self.display_image(self.filtered_frame, self.filtered_label)

    def add_gaussian_noise(self):
        if self.current_frame is None:
            return
        row, col, ch = self.current_frame.shape
        noise_amount = self.noise_amount.value()
        gauss = np.random.normal(0, noise_amount, (row, col, ch)).astype(np.float32)
        noisy = np.clip(self.current_frame + gauss, 0, 255).astype(np.uint8)
        self.filtered_frame = noisy
        self.display_image(self.filtered_frame, self.filtered_label)

    def adjust_hue(self):
        if self.current_frame is None:
            return
        img_hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0].astype(np.float32) + 30) % 180
        self.filtered_frame = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        self.display_image(self.filtered_frame, self.filtered_label)

    def enable_line_drawing(self):
        self.draw_mode = "line"
        self.analysis_info.setText("마우스를 드래그하여 선을 그립니다.")

    def draw_on_image(self, event):
        if self.drawing and self.draw_mode == "line" and self.filtered_frame is not None:
            x = event.pos().x()
            y = event.pos().y()
            label_width = self.filtered_label.width()
            label_height = self.filtered_label.height()
            img_height, img_width = self.filtered_frame.shape[:2]
            scale_x = img_width / label_width
            scale_y = img_height / label_height
            current_point = (int(x * scale_x), int(y * scale_y))
            img = self.filtered_frame.copy()
            cv2.line(img, self.last_point, current_point, (255, 0, 0), 2)
            self.filtered_frame = img
            self.last_point = current_point
            self.display_image(self.filtered_frame, self.filtered_label)

    def draw_rectangle(self):
        if self.filtered_frame is None or self.click_point is None:
            return
        img = self.filtered_frame.copy()
        center = self.click_point
        size = 100
        top_left = (center[0] - size // 2, center[1] - size // 2)
        bottom_right = (center[0] + size // 2, center[1] + size // 2)
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        self.filtered_frame = img
        self.draw_mode = None
        self.display_image(self.filtered_frame, self.filtered_label)

    def add_text(self):
        if self.filtered_frame is None:
            return
        text, ok = QInputDialog.getText(self, "텍스트 입력", "이미지에 추가할 텍스트를 입력하세요:")
        if ok and text:
            img = self.filtered_frame.copy()
            pos = self.click_point if self.click_point else (50, 50)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.filtered_frame = img
            self.display_image(self.filtered_frame, self.filtered_label)

    def random_shapes(self):
        if self.filtered_frame is None:
            return
        img = self.filtered_frame.copy()
        for _ in range(10):
            shape_type = np.random.choice(["circle", "rectangle", "line"])
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            if shape_type == "circle":
                center = (np.random.randint(50, img.shape[1]-50), np.random.randint(50, img.shape[0]-50))
                radius = np.random.randint(10, 50)
                cv2.circle(img, center, radius, color, 2)
            elif shape_type == "rectangle":
                top_left = (np.random.randint(50, img.shape[1]-50), np.random.randint(50, img.shape[0]-50))
                bottom_right = (top_left[0] + 50, top_left[1] + 50)
                cv2.rectangle(img, top_left, bottom_right, color, 2)
            elif shape_type == "line":
                pt1 = (np.random.randint(50, img.shape[1]-50), np.random.randint(50, img.shape[0]-50))
                pt2 = (np.random.randint(50, img.shape[1]-50), np.random.randint(50, img.shape[0]-50))
                cv2.line(img, pt1, pt2, color, 2)
        self.filtered_frame = img
        self.display_image(self.filtered_frame, self.filtered_label)

    def show_histogram(self):
        if self.filtered_frame is None:
            return
        gray = cv2.cvtColor(self.filtered_frame, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist_info = f"히스토그램 데이터 (최대 10개 빈): {hist[:10]}"
        self.analysis_info.setText(hist_info)

    def color_statistics(self):
        if self.filtered_frame is None:
            return
        mean_color = np.mean(self.filtered_frame, axis=(0, 1))
        dominant_color = np.median(self.filtered_frame.reshape(-1, 3), axis=0)
        stats = f"평균 색상 (RGB): {mean_color.astype(int)}, 주요 색상: {dominant_color.astype(int)}"
        self.analysis_info.setText(stats)

    def detect_objects(self):
        if self.filtered_frame is None:
            return
        gray = cv2.cvtColor(self.filtered_frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = self.filtered_frame.copy()
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        self.filtered_frame = img
        self.display_image(self.filtered_frame, self.filtered_label)

    def show_pixel_info(self):
        if self.filtered_frame is None or self.click_point is None:
            return
        self.draw_mode = "pixel_info"
        x, y = self.click_point
        if 0 <= y < self.filtered_frame.shape[0] and 0 <= x < self.filtered_frame.shape[1]:
            pixel = self.filtered_frame[y, x]
            self.analysis_info.setText(f"픽셀 ({x}, {y}) RGB: {pixel}")
        self.draw_mode = None

    def rotate_image(self):
        if self.current_frame is None:
            return
        angle = self.angle_slider.value()
        center = self.click_point if self.click_point else (self.current_frame.shape[1] // 2, self.current_frame.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.current_frame, M, (self.current_frame.shape[1], self.current_frame.shape[0]))
        self.filtered_frame = rotated
        self.display_image(self.filtered_frame, self.filtered_label)

    def set_click_point(self, event):
        """마우스 클릭 위치를 저장하고 필요한 작업을 수행합니다."""
        if self.filtered_frame is None:
            return
        x = event.pos().x()
        y = event.pos().y()
        label_width = self.filtered_label.width()
        label_height = self.filtered_label.height()
        img_height, img_width = self.filtered_frame.shape[:2]
        scale_x = img_width / label_width
        scale_y = img_height / label_height
        self.click_point = (int(x * scale_x), int(y * scale_y))
        
        if self.draw_mode == "line":
            self.last_point = self.click_point
            self.drawing = True
        elif self.draw_mode == "rectangle":
            self.draw_rectangle()
        elif self.draw_mode == "pixel_info":
            self.show_pixel_info()
        elif self.current_filter != "None":
            self.rotate_image()

    def stop_drawing(self, event):
        """그리기 작업을 중지합니다."""
        self.drawing = False
        self.last_point = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AdvancedDetectionApp()
    win.show()
    sys.exit(app.exec_()) 