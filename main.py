import cv2
import mediapipe as mp
import numpy as np
import pybullet as p
import time
import sys
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from pose_analyzer import PoseAnalyzer
from humanoid import HumanoidModel
from training_plan import TrainingPlan
from data_analyzer import DataAnalyzer
from text_renderer import TextRenderer
from pose_model import PosePrediction
from ui.main_window import MainWindow
from voice_feedback import VoiceFeedback  # 添加导入


class FitnessTrainer:
    def __init__(self, user_id="default_user"):
        # 初始化MediaPipe姿势检测
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)

        # 初始化PyBullet（使用DIRECT模式替代GUI模式）
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)

        # 初始化语音反馈
        self.voice_feedback = VoiceFeedback()

        # 初始化姿态分析器和人体模型
        self.pose_analyzer = PoseAnalyzer()
        self.humanoid = HumanoidModel()
        self.humanoid.load_model()

        # 初始化训练计划和数据分析器
        self.user_id = user_id
        self.training_plan = TrainingPlan()
        self.data_analyzer = DataAnalyzer()
        self.current_plan = None
        self.session_data = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'exercises': [],
            'joint_accuracy': {}
        }

        # 初始化文本渲染器
        self.text_renderer = TextRenderer()

        # 将 PoseClassifier 替换为 PyTorch 模型
        self.pose_predictor = PosePrediction()

    def start(self):
        """开始训练"""
        # 加载或创建训练计划
        self.load_or_create_plan()
        # 初始化训练状态
        self.is_running = True
        self.is_paused = False
        self.pause_start_time = None
        self.total_pause_duration = 0

    def pause(self):
        """暂停训练"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_start_time = datetime.now()
        else:
            # 计算暂停时长并累加
            if self.pause_start_time:
                pause_duration = (
                    datetime.now() - self.pause_start_time).seconds
                self.total_pause_duration += pause_duration
                self.pause_start_time = None

    def stop(self):
        """停止训练"""
        self.is_running = False
        self.save_session()
        self.cleanup()

    def process_frame(self, frame=None):
        """处理视频帧
        Args:
            frame: 可选的输入帧，如果为None则从摄像头读取
        Returns:
            处理后的图像帧
        """
        if frame is None:
            success, frame = self.cap.read()
            if not success:
                return None

        # Convert color space and perform pose detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        # Convert back to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Analyze pose and get joint angles
            joint_angles = self.pose_analyzer.analyze_pose(
                results.pose_landmarks)

            # Update PyBullet model pose
            self.humanoid.set_joint_angles(joint_angles)

            # Get target pose from current training plan
            target_angles = self.get_target_angles()
            evaluation = self.pose_analyzer.evaluate_pose(
                joint_angles, target_angles)

            # Detect current pose state
            pose_state = self.pose_analyzer.detect_pose_state(
                results.pose_landmarks)

            # 更新姿势状态历史并检测动作完成情况
            self.track_exercise_completion(pose_state)

            # Update training session data
            self.update_session_data(joint_angles, evaluation)

            # Display pose evaluation results and training progress
            self.draw_evaluation(image, evaluation, pose_state)
            self.draw_training_progress(image)

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return image

    def run(self):
        # Load or create training plan
        self.load_or_create_plan()

        while self.cap.isOpened():
            if not self.process_frame():
                break

            if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
                break

            # Check if training plan is complete
            if self.check_session_complete():
                self.save_session()
                break

        self.cleanup()

    def get_exercise_type(self):
        """Get the exercise type from current training plan"""
        if not self.current_plan or not self.current_plan['exercises']:
            return 'Ready to Start Training'
        current_exercise = self.current_plan['exercises'][0]
        exercise_names = {
            'squat': 'Squat',
            'push_up': 'Push-up',
            'standing_stretch': 'Standing Stretch'
        }
        return exercise_names.get(current_exercise['name'], 'Unknown Exercise')

    def draw_evaluation(self, image, evaluation, pose_state):
        """Display pose evaluation results on image"""
        # Display current exercise type and pose state
        exercise_type = self.get_exercise_type()
        cv2.putText(image, f'Exercise: {exercise_type}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(image, f'Pose State: {pose_state}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Display pose evaluation results
        y_pos = 90
        joint_names = {
            'left_shoulder': 'Left Shoulder',
            'right_shoulder': 'Right Shoulder',
            'left_hip': 'Left Hip',
            'right_hip': 'Right Hip'
        }
        for joint_name, result in evaluation.items():
            status_color = (
                0, 255, 0) if result['status'] == 'good' else (0, 0, 255)
            display_name = joint_names.get(joint_name, joint_name)
            cv2.putText(image, f'{display_name}: {result["suggestion"]}',
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            y_pos += 30

    def cleanup(self):
        # Save final training data
        self.save_session()

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        p.disconnect()

    def load_or_create_plan(self):
        """加载或创建新的训练计划"""
        self.current_plan = self.training_plan.load_plan(self.user_id)
        if not self.current_plan:
            user_info = {
                'user_id': self.user_id,
                'level': 'beginner',
                'goal': 'strength'
            }
            self.current_plan = self.training_plan.create_plan(user_info)

    def get_target_angles(self):
        """获取当前训练动作的目标角度"""
        if not self.current_plan or not self.current_plan['exercises']:
            return {
                'left_shoulder': 0.0,
                'right_shoulder': 0.0,
                'left_hip': 0.0,
                'right_hip': 0.0
            }
        current_exercise = self.current_plan['exercises'][0]
        return current_exercise.get('target_angles', {})

    def update_session_data(self, joint_angles, evaluation):
        """更新训练会话数据"""
        # 更新关节准确度数据
        for joint_name, result in evaluation.items():
            if joint_name not in self.session_data['joint_accuracy']:
                self.session_data['joint_accuracy'][joint_name] = {
                    'accuracy': 0,
                    'count': 0
                }

            joint_data = self.session_data['joint_accuracy'][joint_name]
            accuracy = 100 if result['status'] == 'good' else max(
                0, 100 - abs(result['difference']))
            joint_data['accuracy'] = (
                joint_data['accuracy'] * joint_data['count'] + accuracy) / (joint_data['count'] + 1)
            joint_data['count'] += 1

    def track_exercise_completion(self, pose_state):
        if not self.current_plan or not self.current_plan['exercises']:
            return

        current_exercise = self.current_plan['exercises'][0]
        total_reps = current_exercise.get('reps', 10)  # 获取目标重复次数

        # 初始化姿势状态历史记录
        if not hasattr(self, 'pose_state_history'):
            self.pose_state_history = []
            self.exercise_completed = False
            self.rep_count = 0
            self.last_update_time = datetime.now()

        # 只有当姿势状态变化时才记录和播报
        if not self.pose_state_history or pose_state != self.pose_state_history[-1]:
            self.pose_state_history.append(pose_state)
            # 更新并播报姿势状态
            self.voice_feedback.update_pose_state(pose_state)

            # 检测完整的动作周期
            if len(self.pose_state_history) >= 3:
                # 深蹲完成反馈
                if current_exercise['name'] == 'squat' and pose_state == 'Squat':
                    self.rep_count += 1
                    self.voice_feedback.report_exercise_completion(
                        'squat', self.rep_count, total_reps)
                    self.current_plan['progress']['completed_sessions'] += 1
                    self.pose_state_history = []

                # 俯卧撑完成反馈
                elif current_exercise['name'] == 'push_up':
                    for i in range(len(self.pose_state_history)-2):
                        states = self.pose_state_history[i:i+3]
                        if states[0] == 'Standing' and 'Lying' in states[1:-1] and states[-1] == 'Standing':
                            self.rep_count += 1
                            self.voice_feedback.report_exercise_completion(
                                'push_up', self.rep_count)
                            self.pose_state_history = self.pose_state_history[i+2:]
                            break

            # 限制历史记录长度
            if len(self.pose_state_history) > 10:
                self.pose_state_history = self.pose_state_history[-10:]

            # 检查是否完成了当前动作的所有重复次数
            if self.rep_count >= current_exercise.get('reps', 1):
                self.exercise_completed = True
                self.rep_count = 0  # 重置计数器，准备下一个动作
                self.save_session()
                self.last_update_time = datetime.now()

    def get_session_duration(self):
        """获取实际训练时长（不包括暂停时间）"""
        total_duration = (
            datetime.now() - datetime.fromisoformat(self.session_data['timestamp'])).seconds
        return total_duration - self.total_pause_duration

    def check_session_complete(self):
        """检查训练会话是否完成"""
        if not self.current_plan:
            return False

        # 使用新的计时方法
        session_duration = self.get_session_duration()
        if session_duration >= 1800:  # 30分钟训练时长
            return True

        # 检查是否已完成当前动作
        if hasattr(self, 'exercise_completed') and self.exercise_completed:
            # 如果已经完成了所有动作，返回True
            if not self.current_plan['exercises']:
                return True

            # 如果距离上次更新已经过了一定时间，返回True以触发保存
            if hasattr(self, 'last_update_time') and \
               (datetime.now() - self.last_update_time).seconds > 5:
                return True

        return False

    def save_session(self):
        """保存训练会话数据"""
        if self.session_data['joint_accuracy']:
            self.data_analyzer.save_session_data(
                self.user_id, self.session_data)
            self.training_plan.update_progress(self.user_id, self.session_data)

    def draw_training_progress(self, image):
        """在图像上显示训练进度信息"""
        if not self.current_plan:
            return

        # 显示训练计划信息
        y_pos = 400
        cv2.putText(image, f"Training Plan: Session {self.current_plan['progress']['completed_sessions'] + 1}",
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 显示当前训练时长
        session_duration = self.get_session_duration()
        y_pos += 30
        cv2.putText(image, f"Duration: {session_duration // 60}m {session_duration % 60}s",
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 显示训练完成度
        if self.current_plan['exercises']:
            total_exercises = len(self.current_plan['exercises'])
            completed = self.current_plan['progress']['completed_sessions']
            y_pos += 30
            cv2.putText(image, f"Progress: {completed}/{total_exercises} sets",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    trainer = FitnessTrainer()
    window = MainWindow(trainer)
    window.show()
    sys.exit(app.exec_())
