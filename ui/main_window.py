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

        self.progress_bar.setFormat(
            '%p% - Session Progress')  # Corrected indentation
        progress_layout.addWidget(self.progress_bar)

        self.exercise_label = QLabel('当前动作: 未开始')
        progress_layout.addWidget(self.exercise_label)

        self.rep_label = QLabel('完成次数: 0')
        progress_layout.addWidget(self.rep_label)

        self.feedback_label = QLabel('反馈: 等待开始...')  # 添加反馈标签
        self.feedback_label.setWordWrap(True)
        progress_layout.addWidget(self.feedback_label)

        self.achievement_label = QLabel('成就: 无')  # 添加成就标签
        progress_layout.addWidget(self.achievement_label)

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
            # process_frame now returns frame, evaluation, pose_state
            frame, evaluation, pose_state = self.fitness_trainer.process_frame()
            if frame is not None:
                # 转换颜色空间从 BGR 到 RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 转换为QImage显示
                height, width, channel = rgb_frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(rgb_frame.data, width, height,
                                 bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_image))

                # 更新动作和次数信息
                self.update_exercise_info()

                # 更新反馈标签 (基于 evaluation)
                if evaluation:
                    feedback_msg = self.generate_feedback_message(evaluation)
                    self.update_feedback(feedback_msg)
                elif self.fitness_trainer.is_running and not self.fitness_trainer.is_paused:
                    self.update_feedback("保持姿势...")

                # 更新成就 (示例：完成一个动作)
                if hasattr(self.fitness_trainer, 'rep_count') and self.fitness_trainer.rep_count == 0 and len(self.fitness_trainer.pose_state_history) == 0:
                    # Check if an exercise was just completed (rep_count reset)
                    # This logic might need refinement based on how state is managed
                    pass  # Placeholder for more complex achievement logic

    def generate_feedback_message(self, evaluation):
        """根据评估结果生成用户友好的反馈信息"""
        feedback_parts = []
        has_suggestion = False
        for joint, result in evaluation.items():
            if result['status'] != 'good':
                joint_names = {
                    'left_shoulder': '左肩', 'right_shoulder': '右肩',
                    'left_hip': '左髋', 'right_hip': '右髋',
                    'left_knee': '左膝', 'right_knee': '右膝',
                    'left_elbow': '左肘', 'right_elbow': '右肘'
                    # Add more joints as needed
                }
                joint_name = joint_names.get(joint, joint)
                feedback_parts.append(f"{joint_name}{result['suggestion']}")
                has_suggestion = True

        if not has_suggestion:
            return "姿势标准，继续保持！"
        else:
            return "请注意: " + ", ".join(feedback_parts)

    def update_exercise_info(self):
        if hasattr(self.fitness_trainer, 'current_plan') and self.fitness_trainer.current_plan:
            plan = self.fitness_trainer.current_plan
            total_exercises = len(
                plan.get('initial_exercises', plan['exercises']))  # 获取初始总数
            current_exercise_index = total_exercises - len(plan['exercises'])

            if plan['exercises']:
                exercise = plan['exercises'][0]
                self.exercise_label.setText(
                    f"当前动作: {exercise['name']} ({current_exercise_index + 1}/{total_exercises})")
                if hasattr(self.fitness_trainer, 'rep_count'):
                    target_reps = exercise.get('reps', '-')
                    self.rep_label.setText(
                        f"完成次数: {self.fitness_trainer.rep_count} / {target_reps}")

                # 更新进度条 (基于完成的动作数)
                progress = (current_exercise_index / total_exercises) * \
                    100 if total_exercises > 0 else 0
                self.progress_bar.setValue(int(progress))
            else:
                self.exercise_label.setText("训练完成!")
                self.rep_label.setText("-")
                self.progress_bar.setValue(100)

    def update_feedback(self, feedback_text):
        """更新反馈标签内容"""
        self.feedback_label.setText(f"反馈: {feedback_text}")

    def update_achievements(self, achievement_text):
        """更新成就标签内容"""
        self.achievement_label.setText(f"成就: {achievement_text}")

    # 在 start_training 中记录初始练习列表
    def start_training(self):
        # ... (获取用户设置)
        level_map = {'初学者': 'beginner', '中级': 'intermediate', '高级': 'advanced'}
        goal_map = {'力量训练': 'strength',
                    '灵活性训练': 'flexibility', '有氧运动': 'cardio'}

        user_info = {
            'user_id': self.fitness_trainer.user_id,
            'level': level_map[self.level_combo.currentText()],
            'goal': goal_map[self.goal_combo.currentText()]
        }

        # 获取分析数据并创建计划
        analysis_data = self.fitness_trainer.data_analyzer.analyze_progress(
            self.fitness_trainer.user_id)
        self.fitness_trainer.current_plan = self.fitness_trainer.training_plan.create_plan(
            user_info, analysis_data)
        # 记录初始练习列表以计算总进度
        if self.fitness_trainer.current_plan:
            self.fitness_trainer.current_plan['initial_exercises'] = self.fitness_trainer.current_plan['exercises'].copy(
            )

        self.fitness_trainer.start_session()  # 改为调用 start_session
        self.timer.start(30)
        self.update_exercise_info()
        self.update_feedback("训练开始!")  # 初始反馈
        self.update_achievements("开始新的挑战!")

    def pause_training(self):
        """暂停或恢复训练"""
        if not self.fitness_trainer.is_running:
            return  # 训练未开始，不执行任何操作

        if self.fitness_trainer.is_paused:
            # 恢复训练
            self.fitness_trainer.resume()
            self.pause_button.setText('暂停')
            self.timer.start(30)
            self.update_feedback("训练已恢复")
        else:
            # 暂停训练
            self.fitness_trainer.pause()
            self.pause_button.setText('恢复')
            self.timer.stop()
            self.update_feedback("训练已暂停")

    def stop_training(self):
        """停止当前训练会话"""
        self.fitness_trainer.stop()
        self.timer.stop()
        self.update_exercise_info()  # 更新UI显示训练结束
        self.update_feedback("训练已结束")
        self.update_achievements("完成本次训练!")
        self.progress_bar.setValue(0)  # 重置进度条
        self.exercise_label.setText('当前动作: 未开始')
        self.rep_label.setText('完成次数: 0')
        self.pause_button.setText('暂停')  # 重置暂停按钮文本

    # 在 process_frame 中更新UI反馈 (需要修改 main.py 以调用)
    # def update_frame(self):
    #     if hasattr(self.fitness_trainer, 'cap') and self.fitness_trainer.cap.isOpened():
    #         frame, evaluation, pose_state = self.fitness_trainer.process_frame() # 假设 process_frame 返回更多信息
    #         if frame is not None:
    #             # ... (图像显示代码)
    #             self.update_exercise_info() # 更新动作和次数
    #             # 更新反馈标签 (基于 evaluation)
    #             feedback_msg = self.generate_feedback_message(evaluation)
    #             self.update_feedback(feedback_msg)
    #             # 更新成就 (可以基于进度或特定表现)
    #             # self.update_achievements(...)
