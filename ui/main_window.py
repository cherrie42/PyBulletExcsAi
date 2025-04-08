import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QComboBox, 
                           QProgressBar, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2

class MainWindow(QMainWindow):
    def __init__(self, fitness_trainer):
        super().__init__()
        self.fitness_trainer = fitness_trainer
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('AI 健身教练')
        self.setMinimumSize(1200, 800)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        layout = QHBoxLayout(main_widget)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 右侧视频显示区域
        video_panel = self.create_video_panel()
        layout.addWidget(video_panel)
        
        # 设置定时器更新视频帧
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 刷新一次
        
    def create_control_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.Box)
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        
        # 用户信息
        user_group = QFrame()
        user_layout = QVBoxLayout(user_group)
        user_layout.addWidget(QLabel('训练计划设置'))
        
        # 难度选择
        self.level_combo = QComboBox()
        self.level_combo.addItems(['初学者', '中级', '高级'])
        user_layout.addWidget(QLabel('难度级别:'))
        user_layout.addWidget(self.level_combo)
        
        # 训练目标选择
        self.goal_combo = QComboBox()
        self.goal_combo.addItems(['力量训练', '灵活性训练', '有氧运动'])
        user_layout.addWidget(QLabel('训练目标:'))
        user_layout.addWidget(self.goal_combo)
        
        layout.addWidget(user_group)
        
        # 训练进度
        progress_group = QFrame()
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.addWidget(QLabel('训练进度'))
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.exercise_label = QLabel('当前动作: 未开始')
        progress_layout.addWidget(self.exercise_label)
        
        self.rep_label = QLabel('完成次数: 0')
        progress_layout.addWidget(self.rep_label)
        
        layout.addWidget(progress_group)
        
        # 控制按钮
        button_group = QFrame()
        button_layout = QVBoxLayout(button_group)
        
        self.start_button = QPushButton('开始训练')
        self.start_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton('暂停')
        self.pause_button.clicked.connect(self.pause_training)
        button_layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton('结束训练')
        self.stop_button.clicked.connect(self.stop_training)
        button_layout.addWidget(self.stop_button)
        
        layout.addWidget(button_group)
        layout.addStretch()
        
        return panel
        
    def create_video_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.Box)
        layout = QVBoxLayout(panel)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        return panel
        
    def update_frame(self):
        if hasattr(self.fitness_trainer, 'cap') and self.fitness_trainer.cap.isOpened():
            frame = self.fitness_trainer.process_frame()
            if frame is not None:
                # 转换颜色空间从 BGR 到 RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 转换为QImage显示
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_image))
                
    def start_training(self):
        # 获取用户设置
        level_map = {'初学者': 'beginner', '中级': 'intermediate', '高级': 'advanced'}
        goal_map = {'力量训练': 'strength', '灵活性训练': 'flexibility', '有氧运动': 'cardio'}
        
        user_info = {
            'user_id': self.fitness_trainer.user_id,  # 添加 user_id
            'level': level_map[self.level_combo.currentText()],
            'goal': goal_map[self.goal_combo.currentText()]
        }
        
        # 更新训练计划
        self.fitness_trainer.current_plan = self.fitness_trainer.training_plan.create_plan(user_info)
        self.fitness_trainer.start()
        self.timer.start(30)  # 确保定时器启动
        self.update_exercise_info()
        
    def pause_training(self):
        self.fitness_trainer.pause()
        if self.fitness_trainer.is_paused:
            self.timer.stop()  # 暂停定时器
            self.pause_button.setText('继续')
        else:
            self.timer.start(30)  # 重新启动定时器
            self.pause_button.setText('暂停')
        
    def stop_training(self):
        self.fitness_trainer.stop()
        self.close()
        
    def update_exercise_info(self):
        if hasattr(self.fitness_trainer, 'current_plan'):
            plan = self.fitness_trainer.current_plan
            if plan and plan['exercises']:
                exercise = plan['exercises'][0]
                self.exercise_label.setText(f"当前动作: {exercise['name']}")
                if hasattr(self.fitness_trainer, 'rep_count'):
                    self.rep_label.setText(f"完成次数: {self.fitness_trainer.rep_count}")
                
                # 更新进度条
                if 'progress' in plan:
                    progress = (plan['progress']['completed_sessions'] / 
                              len(plan['exercises']) * 100)
                    self.progress_bar.setValue(int(progress))