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
from text_renderer import TextRenderer  # 在文件开头的导入部分添加

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
        
        # 初始化文本渲染器
        self.text_renderer = TextRenderer()  # 添加这一行
    def process_frame(self):
        success, image = self.cap.read()
        if not success:
            return False
            
        # Convert color space and perform pose detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        # Convert back to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            # Analyze pose and get joint angles
            joint_angles = self.pose_analyzer.analyze_pose(results.pose_landmarks)
            
            # Update PyBullet model pose
            self.humanoid.set_joint_angles(joint_angles)
            
            # Get target pose from current training plan
            target_angles = self.get_target_angles()
            evaluation = self.pose_analyzer.evaluate_pose(joint_angles, target_angles)
            
            # Detect current pose state
            pose_state = self.pose_analyzer.detect_pose_state(results.pose_landmarks)
            
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
            
        # Display processed image
        cv2.imshow('Fitness Trainer', image)
        return True
        
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
            status_color = (0, 255, 0) if result['status'] == 'good' else (0, 0, 255)
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