import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import pybullet as p
import time
import json

class TrainingDataGenerator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 初始化保存路径
        self.data_dir = Path('training_data/pose_images')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建运动类型目录
        for exercise in ['squat', 'push_up', 'standing_stretch']:
            (self.data_dir / exercise).mkdir(exist_ok=True)
            
        # 初始化 PyBullet
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        
        # 加载人体模型
        self.humanoid = p.loadURDF("models/humanoid.urdf")
        
    def capture_standard_poses(self):
        """捕获标准姿势数据"""
        cap = cv2.VideoCapture(1)
        
        exercise_types = {
            's': 'squat',
            'p': 'push_up',
            't': 'standing_stretch'
        }
        
        print("准备捕获标准姿势数据:")
        print("按 's' 捕获深蹲动作")
        print("按 'p' 捕获俯卧撑动作")
        print("按 't' 捕获站姿拉伸动作")
        print("按 'q' 退出")
        
        frame_count = 0
        current_exercise = None
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
                
            # 处理图像
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            # 显示骨架
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
            
            # 显示当前状态
            if current_exercise:
                cv2.putText(image_bgr, f"Recording: {current_exercise}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.imshow('Standard Pose Capture', image_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 处理按键
            if key == ord('q'):
                break
            elif chr(key) in exercise_types:
                current_exercise = exercise_types[chr(key)]
                frame_count = 0
                print(f"开始记录 {current_exercise} 动作")
            elif key == ord('c') and current_exercise:
                # 保存当前帧
                if results.pose_landmarks:
                    frame_count += 1
                    image_path = self.data_dir / current_exercise / f"{frame_count}.jpg"
                    cv2.imwrite(str(image_path), image_bgr)
                    
                    # 保存关键点数据
                    landmarks_data = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks_data.extend([landmark.x, landmark.y, landmark.z])
                    
                    json_path = image_path.with_suffix('.json')
                    with open(json_path, 'w') as f:
                        json.dump({
                            'exercise_type': current_exercise,
                            'landmarks': landmarks_data
                        }, f)
                    
                    print(f"已保存第 {frame_count} 帧 {current_exercise} 动作数据")
        
        cap.release()
        cv2.destroyAllWindows()

    def generate_synthetic_data(self):
        """生成合成训练数据"""
        exercise_angles = {
            'squat': [
                # 起始姿势
                {'hip': 0, 'knee': 0},
                # 下蹲过程
                {'hip': -45, 'knee': 45},
                # 完全下蹲
                {'hip': -90, 'knee': 90}
            ],
            'push_up': [
                # 起始姿势
                {'shoulder': 0, 'elbow': 0},
                # 下降过程
                {'shoulder': -45, 'elbow': 45},
                # 最低点
                {'shoulder': -90, 'elbow': 90}
            ],
            'standing_stretch': [
                # 基本站姿
                {'shoulder': 0, 'elbow': 0},
                # 伸展
                {'shoulder': 90, 'elbow': 0},
                # 完全伸展
                {'shoulder': 180, 'elbow': 0}
            ]
        }
        
        for exercise_type, poses in exercise_angles.items():
            print(f"生成 {exercise_type} 训练数据...")
            for i, pose in enumerate(poses):
                # 设置关节角度
                self._set_joint_angles(pose)
                time.sleep(0.5)  # 等待物理引擎稳定
                
                # 获取关节位置
                joint_positions = self._get_joint_positions()
                
                # 保存数据
                self._save_pose_data(exercise_type, i, joint_positions)
                
    def _set_joint_angles(self, angles):
        """设置人体模型关节角度"""
        for joint_name, angle in angles.items():
            if joint_name == 'hip':
                p.setJointMotorControl2(self.humanoid, 0, p.POSITION_CONTROL, 
                                      targetPosition=np.radians(angle))
            elif joint_name == 'knee':
                p.setJointMotorControl2(self.humanoid, 1, p.POSITION_CONTROL, 
                                      targetPosition=np.radians(angle))
            elif joint_name == 'shoulder':
                p.setJointMotorControl2(self.humanoid, 2, p.POSITION_CONTROL, 
                                      targetPosition=np.radians(angle))
            elif joint_name == 'elbow':
                p.setJointMotorControl2(self.humanoid, 3, p.POSITION_CONTROL, 
                                      targetPosition=np.radians(angle))
                
    def _get_joint_positions(self):
        """获取关节位置"""
        positions = []
        for i in range(p.getNumJoints(self.humanoid)):
            joint_info = p.getJointInfo(self.humanoid, i)
            joint_state = p.getJointState(self.humanoid, i)
            pos = p.getLinkState(self.humanoid, i)[0]
            positions.extend(pos)
        return positions
        
    def _save_pose_data(self, exercise_type, index, positions):
        """保存姿势数据"""
        data_path = self.data_dir / exercise_type / f"synthetic_{index}.json"
        data = {
            'exercise_type': exercise_type,
            'landmarks': positions
        }
        
        with open(data_path, 'w') as f:
            json.dump(data, f)
            
        print(f"已保存 {exercise_type} 姿势 {index}")

if __name__ == '__main__':
    generator = TrainingDataGenerator()
    # 生成合成数据
    generator.generate_synthetic_data()
    # 捕获真实数据
    generator.capture_standard_poses()