import pybullet as p
import numpy as np
from pathlib import Path

class HumanoidModel:
    def __init__(self):
        # Load URDF model
        self.model_path = str(Path(__file__).parent / 'models' / 'humanoid.urdf')
        self.humanoid_id = None
        self.joint_indices = []
        self.joint_names = []
        
    def load_model(self):
        """Load humanoid model into PyBullet environment"""
        self.humanoid_id = p.loadURDF(
            self.model_path,
            basePosition=[0, 0, 1],
            useFixedBase=True
        )
        
        # Get joint information
        for i in range(p.getNumJoints(self.humanoid_id)):
            joint_info = p.getJointInfo(self.humanoid_id, i)
            self.joint_indices.append(joint_info[0])
            self.joint_names.append(joint_info[1].decode('utf-8'))
            
    def set_joint_angles(self, joint_angles):
        """Set joint angles
        Args:
            joint_angles: Dictionary mapping joint names to angle values
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
        """Get current angles of all joints"""
        joint_states = {}
        for i, name in enumerate(self.joint_names):
            state = p.getJointState(self.humanoid_id, i)
            joint_states[name] = state[0]
        return joint_states
    
    def reset_pose(self):
        """Reset humanoid model to initial pose"""
        for joint_index in self.joint_indices:
            p.resetJointState(self.humanoid_id, joint_index, 0)
            
    def calculate_pose_difference(self, target_angles):
        """Calculate difference between current pose and target pose
        Args:
            target_angles: Dictionary of target joint angles
        Returns:
            Dictionary of angle differences for each joint
        """
        current_angles = self.get_joint_angles()
        differences = {}
        for joint_name in self.joint_names:
            if joint_name in target_angles:
                diff = target_angles[joint_name] - current_angles[joint_name]
                differences[joint_name] = np.degrees(diff)  # Convert to degrees
        return differences