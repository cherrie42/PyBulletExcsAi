import pybullet as p
import numpy as np
from pathlib import Path

class HumanoidModel:
    def __init__(self):
        # 加载URDF模型
        self.model_path = str(Path(__file__).parent / 'models' / 'humanoid.urdf')
        self.humanoid_id = None
        self.joint_indices = []
        self.joint_names = []
        
    def load_model(self):
        """加载人体模型到PyBullet环境中"""
        self.humanoid_id = p.loadURDF(
            self.model_path,
            basePosition=[0, 0, 1],
            useFixedBase=True
        )
        
        # 获取关节信息
        for i in range(p.getNumJoints(self.humanoid_id)):
            joint_info = p.getJointInfo(self.humanoid_id, i)
            self.joint_indices.append(joint_info[0])
            self.joint_names.append(joint_info[1].decode('utf-8'))
            
    def set_joint_angles(self, joint_angles):
        """设置关节角度
        Args:
            joint_angles: 字典，关节名称到角度值的映射
        """
        for joint_name, angle in joint_angles.items():
            if joint_name in self.joint_names:
                joint_index = self.joint_names.index(joint_name)
                p.setJointMotorControl2(
                    self.humanoid_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=angle
                )
                
    def get_joint_angles(self):
        """获取当前所有关节角度"""
        joint_states = {}
        for i, name in enumerate(self.joint_names):
            state = p.getJointState(self.humanoid_id, i)
            joint_states[name] = state[0]
        return joint_states
    
    def reset_pose(self):
        """重置人体模型到初始姿态"""
        for joint_index in self.joint_indices:
            p.resetJointState(self.humanoid_id, joint_index, 0)
            
    def calculate_pose_difference(self, target_angles):
        """计算当前姿态与目标姿态的差异
        Args:
            target_angles: 字典，目标关节角度
        Returns:
            字典，每个关节的角度差异
        """
        current_angles = self.get_joint_angles()
        differences = {}
        for joint_name in self.joint_names:
            if joint_name in target_angles:
                diff = target_angles[joint_name] - current_angles[joint_name]
                differences[joint_name] = np.degrees(diff)  # 转换为角度
        return differences