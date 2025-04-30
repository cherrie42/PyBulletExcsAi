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
        self.data_analyzer = DataAnalyzer()  # DataAnalyzer 实例
        self.current_plan = None
        self.session_data = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'exercises': [],
            'joint_accuracy': {},
            'rep_counts': {},
            'pose_states': []
        }
        self.rep_count = 0  # 初始化动作计数
        self.pose_state_history = []  # 初始化姿势状态历史

        # 初始化文本渲染器
        self.text_renderer = TextRenderer()

        # 将 PoseClassifier 替换为 PyTorch 模型
        self.pose_predictor = PosePrediction()

        # 初始化运行和暂停状态
        self.is_running = False
        self.is_paused = False

    def start_session(self):
        """开始训练"""
        # 加载或创建训练计划，并传入分析数据
        analysis_data = self.data_analyzer.analyze_progress(self.user_id)
        self.current_plan = self.training_plan.create_plan(
            {'user_id': self.user_id, 'level': 'beginner', 'goal': 'strength'}, analysis_data)
        # 初始化训练状态
        self.is_running = True
        self.is_paused = False
        self.pause_start_time = None
        self.total_pause_duration = 0
        self.rep_count = 0  # 重置计数器
        self.pose_state_history = []  # 重置历史
        self.voice_feedback.announce_session_status('start')
        # 移除原有的 while 循环，帧处理由 UI 的 update_frame 调用 process_frame 驱动

    def process_frame(self):
        """处理单帧图像，进行姿势检测和分析"""
        if not self.is_running or self.is_paused:
            # 如果训练未运行或已暂停，尝试读取帧以保持摄像头活动，但不处理
            ret, frame = self.cap.read()
            if not ret:
                return None, None, None  # 无法读取帧
            # 可以选择返回原始帧或一个表示暂停/未运行状态的帧
            # 这里返回原始帧，但没有评估和姿势状态
            return frame, None, "Paused/Stopped"

        ret, frame = self.cap.read()
        if not ret:
            print("无法从摄像头读取帧")
            self.is_running = False  # 停止运行
            return None, None, None

        # 转换颜色空间 BGR -> RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 进行姿势检测
        results = self.pose.process(image)

        # 转换颜色空间 RGB -> BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        evaluation = None
        pose_state = "Detecting..."

        if results.pose_landmarks:
            # 绘制姿势关键点
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # 获取关键点坐标
            landmarks = results.pose_landmarks.landmark

            # 进行姿态分析
            joint_angles = self.pose_analyzer.analyze_pose(
                results.pose_landmarks)  # 使用 analyze_pose 并传入 pose_landmarks
            target_angles = self.get_target_angles()
            evaluation = self.pose_analyzer.evaluate_pose(
                joint_angles, target_angles)

            # 使用 PyTorch 模型预测姿势状态
            pose_state = self.pose_predictor.predict(landmarks)

            # 更新人体模型姿态
            self.humanoid.update_pose(joint_angles)

            # 跟踪运动完成情况
            self.track_exercise_completion(pose_state, landmarks)

            # 更新会话数据
            self.update_session_data(joint_angles, evaluation, pose_state)

            # 在图像上绘制评估结果
            self.draw_evaluation(image, evaluation, pose_state)

            # 检查训练计划是否完成 (可以在这里检查，或者由UI触发)
            if self.check_session_complete():
                self.save_session()
                self.is_running = False  # 标记为停止
                # 可以通知 UI 训练已完成
                print("训练计划完成!")
                # 注意：这里直接停止可能不理想，最好由UI控制停止流程

        else:
            # 如果未检测到姿势，显示提示信息
            cv2.putText(image, "No pose detected", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pose_state = "No pose detected"
            evaluation = {}  # 返回空评估

        return image, evaluation, pose_state

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
        if evaluation:  # 确保 evaluation 不是 None
            y_pos = 90
            joint_names = {
                'left_shoulder': 'Left Shoulder',
                'right_shoulder': 'Right Shoulder',
                'left_hip': 'Left Hip',
                'right_hip': 'Right Hip',
                'left_knee': 'Left Knee',  # 添加更多关节
                'right_knee': 'Right Knee',
                'left_elbow': 'Left Elbow',
                'right_elbow': 'Right Elbow'
            }
            for joint_name, result in evaluation.items():
                status_color = (
                    0, 255, 0) if result['status'] == 'good' else (0, 0, 255)
                display_name = joint_names.get(joint_name, joint_name)
                cv2.putText(image, f'{display_name}: {result["suggestion"]}',
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                y_pos += 30

    def cleanup(self):
        # Save final training data if the session was running
        # 检查 self.is_running 状态可能不再准确，因为停止可能由UI触发
        # 考虑在 stop_training 或窗口关闭时保存
        # if self.is_running: # 移除此检查，总是在清理时尝试保存
        self.save_session()

        # Release resources
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()  # 这一行可能需要移除，因为窗口由 PyQt 管理
        # Check if PyBullet is connected before disconnecting
        if p.isConnected():
            p.disconnect()
        print("资源已清理")

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
            # 返回默认值或引发错误
            print("Warning: No current plan or exercises found for target angles.")
            return {
                'left_shoulder': 90.0, 'right_shoulder': 90.0,
                'left_hip': 180.0, 'right_hip': 180.0,
                'left_knee': 180.0, 'right_knee': 180.0,
                'left_elbow': 180.0, 'right_elbow': 180.0
            }
        current_exercise = self.current_plan['exercises'][0]
        # 提供默认空字典，以防 'target_angles' 不存在
        return current_exercise.get('target_angles', {})

    def update_session_data(self, joint_angles, evaluation, pose_state):
        """更新训练会话数据"""
        if not evaluation:  # 如果没有评估数据，则不更新
            return

        timestamp = datetime.now().isoformat()
        current_exercise = self.current_plan['exercises'][0][
            'name'] if self.current_plan and self.current_plan['exercises'] else 'None'

        # 记录每个关节的准确度
        for joint, result in evaluation.items():
            if joint not in self.session_data['joint_accuracy']:
                self.session_data['joint_accuracy'][joint] = {
                    'timestamps': [], 'accuracies': []}
            self.session_data['joint_accuracy'][joint]['timestamps'].append(
                timestamp)
            # 假设准确度是 100 (good) 或 0 (bad)
            accuracy = 100 if result['status'] == 'good' else 0
            self.session_data['joint_accuracy'][joint]['accuracies'].append(
                accuracy)

        # 记录姿势状态
        self.session_data['pose_states'].append(
            {'timestamp': timestamp, 'state': pose_state})

        # 记录当前动作和次数 (如果适用)
        if current_exercise != 'None':
            if current_exercise not in self.session_data['rep_counts']:
                self.session_data['rep_counts'][current_exercise] = 0
            # 更新当前次数
            self.session_data['rep_counts'][current_exercise] = self.rep_count

    def track_exercise_completion(self, current_pose_state, landmarks):
        """跟踪运动完成情况，使用 landmarks 进行更精确判断"""
        if not self.current_plan or not self.current_plan['exercises']:
            return

        current_exercise = self.current_plan['exercises'][0]
        ex_name = current_exercise['name']
        target_reps = current_exercise.get('reps')

        # 状态转换检测逻辑 (简化示例)
        last_state = self.pose_state_history[-1] if self.pose_state_history else None

        # 简单的状态转换计数逻辑 (需要根据具体动作调整)
        # 例如：深蹲，从 'up' -> 'down' -> 'up' 算一次
        rep_completed = False
        if ex_name == 'squat':
            if last_state == 'down' and current_pose_state == 'up':
                rep_completed = True
        elif ex_name == 'push_up':
            if last_state == 'down' and current_pose_state == 'up':  # 假设俯卧撑也是 down -> up
                rep_completed = True
        # 可以为其他动作添加类似逻辑

        # 更新历史记录
        self.pose_state_history.append(current_pose_state)
        if len(self.pose_state_history) > 5:  # 保留最近5个状态用于检测
            self.pose_state_history.pop(0)

        if rep_completed:
            self.rep_count += 1
            print(f"{ex_name} 完成次数: {self.rep_count}")
            self.voice_feedback.announce_rep_count(self.rep_count)

            # 检查是否达到目标次数
            if target_reps and self.rep_count >= target_reps:
                print(f"动作 {ex_name} 完成!")
                self.voice_feedback.announce_exercise_completion(ex_name)
                self.session_data['exercises'].append({
                    'name': ex_name,
                    'reps_completed': self.rep_count,
                    'target_reps': target_reps,
                    'completion_time': datetime.now().isoformat()
                })
                # 切换到下一个动作
                self.current_plan['exercises'].pop(0)
                self.rep_count = 0  # 重置计数器
                self.pose_state_history = []  # 重置历史
                if not self.current_plan['exercises']:
                    print("所有动作完成!")
                    # 可以在这里触发训练完成的逻辑
                else:
                    next_exercise = self.current_plan['exercises'][0]['name']
                    print(f"下一个动作: {next_exercise}")
                    self.voice_feedback.announce_next_exercise(next_exercise)

    def check_session_complete(self):
        """检查整个训练计划是否完成"""
        return self.current_plan and not self.current_plan['exercises']

    def save_session(self):
        """保存当前训练会话数据"""
        if not self.session_data or not self.session_data.get('exercises'):
            # 如果没有有效的训练数据（例如，没有完成任何动作），则不保存
            print("没有有效的训练数据可保存。")
            return

        # 确保保存目录存在
        save_dir = Path('data') / self.user_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = save_dir / f"session_{timestamp_str}.json"

        # 保存数据
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                import json
                json.dump(self.session_data, f, ensure_ascii=False, indent=4)
            print(f"训练数据已保存到 {filename}")
        except Exception as e:
            print(f"保存训练数据时出错: {e}")

    def pause_resume(self):
        """暂停或恢复训练"""
        if not self.is_running:
            return  # 训练未开始，无法暂停/恢复

        if self.is_paused:
            # 恢复训练
            self.is_paused = False
            pause_duration = time.time() - self.pause_start_time
            self.total_pause_duration += pause_duration
            print(f"训练已恢复，暂停时长: {pause_duration:.2f} 秒")
            self.voice_feedback.announce_session_status('resume')
        else:
            # 暂停训练
            self.is_paused = True
            self.pause_start_time = time.time()
            print("训练已暂停")
            self.voice_feedback.announce_session_status('pause')

    def stop(self):
        """停止训练"""
        if self.is_running:
            self.is_running = False
            print("训练已停止")
            self.voice_feedback.announce_session_status('stop')
            self.cleanup()  # 停止时进行清理和保存
        else:
            print("训练尚未开始或已停止")


# 主程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 创建 FitnessTrainer 实例
    trainer = FitnessTrainer(user_id="test_user_123")

    # 创建主窗口并传入 trainer
    main_win = MainWindow(trainer)
    main_win.show()

    sys.exit(app.exec_())
