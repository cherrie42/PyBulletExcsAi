import cv2
import mediapipe as mp
import numpy as np
import pybullet as p
import time
from datetime import datetime
from pathlib import Path
from pose_analyzer import PoseAnalyzer
from humanoid import HumanoidModel
from training_plan import TrainingPlan
from data_analyzer import DataAnalyzer
from text_renderer import TextRenderer

class FitnessTrainer:
    def __init__(self, user_id="default_user"):
        # 初始化MediaPipe姿势检测
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        
        # 初始化PyBullet
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        
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
        
    def process_frame(self):
        success, image = self.cap.read()
        if not success:
            return False
            
        # 转换颜色空间并进行姿势检测
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        # 转回BGR用于显示
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            # 分析姿态并获取关节角度
            joint_angles = self.pose_analyzer.analyze_pose(results.pose_landmarks)
            
            # 更新PyBullet模型姿态
            self.humanoid.set_joint_angles(joint_angles)
            
            # 获取当前训练计划中的目标姿态
            target_angles = self.get_target_angles()
            evaluation = self.pose_analyzer.evaluate_pose(joint_angles, target_angles)
            
            # 检测当前姿态状态
            pose_state = self.pose_analyzer.detect_pose_state(results.pose_landmarks)
            
            # 更新训练会话数据
            self.update_session_data(joint_angles, evaluation)
            
            # 在图像上显示姿态评估结果和训练进度
            self.draw_evaluation(image, evaluation, pose_state)
            self.draw_training_progress(image)
            
            # 绘制姿势标记点
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
        # 显示处理后的图像
        cv2.imshow('Fitness Trainer', image)
        return True
        
    def run(self):
        # 加载或创建训练计划
        self.load_or_create_plan()
        
        while self.cap.isOpened():
            if not self.process_frame():
                break
                
            if cv2.waitKey(5) & 0xFF == 27:  # ESC键退出
                break
            
            # 检查训练计划完成情况
            if self.check_session_complete():
                self.save_session()
                break
                
        self.cleanup()
        
    def get_exercise_type(self):
        """根据当前训练计划识别运动类型"""
        if not self.current_plan or not self.current_plan['exercises']:
            return '准备开始训练'
        current_exercise = self.current_plan['exercises'][0]
        exercise_names = {
            'squat': '深蹲',
            'push_up': '俯卧撑',
            'standing_stretch': '站姿拉伸'
        }
        return exercise_names.get(current_exercise['name'], '未知运动')

    def draw_evaluation(self, image, evaluation, pose_state):
        """在图像上显示姿态评估结果"""
        # 显示当前运动类型和姿态状态
        exercise_type = self.get_exercise_type()
        image = self.text_renderer.put_text(image, f'当前运动: {exercise_type}',
                    (10, 30), font_size=24, color=(255, 255, 0))
        image = self.text_renderer.put_text(image, f'姿态状态: {pose_state}',
                    (10, 60), font_size=24, color=(0, 255, 255))

        # 显示姿态评估结果
        y_pos = 90
        joint_names = {
            'left_shoulder': '左肩',
            'right_shoulder': '右肩',
            'left_hip': '左髋',
            'right_hip': '右髋'
        }
        for joint_name, result in evaluation.items():
            status_color = (0, 255, 0) if result['status'] == 'good' else (0, 0, 255)
            display_name = joint_names.get(joint_name, joint_name)
            image = self.text_renderer.put_text(image, f'{display_name}: {result["suggestion"]}',
                        (10, y_pos), font_size=24, color=status_color)
            y_pos += 30
    
    def cleanup(self):
        # 保存最终的训练数据
        self.save_session()
        
        # 释放资源
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
            accuracy = 100 if result['status'] == 'good' else max(0, 100 - abs(result['difference']))
            joint_data['accuracy'] = (joint_data['accuracy'] * joint_data['count'] + accuracy) / (joint_data['count'] + 1)
            joint_data['count'] += 1
            
    def check_session_complete(self):
        """检查训练会话是否完成"""
        if not self.current_plan:
            return False
            
        # 检查是否达到计划的训练时长或完成所有动作
        session_duration = (datetime.now() - datetime.fromisoformat(self.session_data['timestamp'])).seconds
        return session_duration >= 1800  # 30分钟训练时长
        
    def save_session(self):
        """保存训练会话数据"""
        if self.session_data['joint_accuracy']:
            self.data_analyzer.save_session_data(self.user_id, self.session_data)
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
        session_duration = (datetime.now() - datetime.fromisoformat(self.session_data['timestamp'])).seconds
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
    trainer = FitnessTrainer()
    trainer.run()