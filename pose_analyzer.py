import numpy as np
import mediapipe as mp

class PoseAnalyzer:
    def __init__(self):
        # 定义关键点到关节角度的映射关系
        self.joint_mapping = {
            'left_shoulder': [11, 13, 15],  # 左肩、左肘、左腕
            'right_shoulder': [12, 14, 16],  # 右肩、右肘、右腕
            'left_hip': [23, 25, 27],  # 左髋、左膝、左踝
            'right_hip': [24, 26, 28],  # 右髋、右膝、右踝
        }
        
        # 定义姿态状态判断的阈值
        self.pose_thresholds = {
            'standing': {
                'hip_knee_angle': 160,  # 站立时髋膝角度阈值（接近180度）
                'vertical_ratio': 0.8    # 站立时身高比例阈值
            },
            'sitting': {
                'hip_knee_angle': 110,   # 坐姿时髋膝角度阈值（约90-120度）
                'vertical_ratio': 0.6    # 坐姿时身高比例阈值
            }
        }
        
    def calculate_angle(self, a, b, c):
        """计算三个点形成的角度
        Args:
            a, b, c: 三个点的3D坐标，b是角度的顶点
        Returns:
            角度值（弧度）
        """
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return angle
    
    def analyze_pose(self, pose_landmarks):
        """分析姿态数据，计算关节角度
        Args:
            pose_landmarks: MediaPipe姿态检测结果
        Returns:
            字典，包含各关节的角度值
        """
        if not pose_landmarks:
            return {}
            
        joint_angles = {}
        landmarks = pose_landmarks.landmark
        
        for joint_name, (p1, p2, p3) in self.joint_mapping.items():
            angle = self.calculate_angle(
                landmarks[p1],
                landmarks[p2],
                landmarks[p3]
            )
            joint_angles[joint_name] = angle
            
        return joint_angles
        
    def evaluate_pose(self, current_angles, target_angles, threshold=0.1):
        """评估当前姿态与目标姿态的差异
        Args:
            current_angles: 当前姿态的关节角度
            target_angles: 目标姿态的关节角度
            threshold: 角度差异阈值（弧度）
        Returns:
            评估结果，包含每个关节的状态和建议
        """
        evaluation = {}
        for joint_name in current_angles:
            if joint_name in target_angles:
                diff = abs(current_angles[joint_name] - target_angles[joint_name])
                status = 'good' if diff < threshold else 'needs_correction'
                
                evaluation[joint_name] = {
                    'status': status,
                    'difference': np.degrees(diff),
                    'suggestion': self._get_correction_suggestion(
                        joint_name,
                        current_angles[joint_name],
                        target_angles[joint_name]
                    ) if status == 'needs_correction' else 'Correct posture, please maintain'
                }
                
        return evaluation
        
    def _get_correction_suggestion(self, joint_name, current_angle, target_angle):
        diff = np.degrees(current_angle - target_angle)
        direction = 'bend' if diff < 0 else 'extend'
        
        suggestions = {
            'left_shoulder': f'Left shoulder needs to {direction} {abs(diff):.1f} degrees',
            'right_shoulder': f'Right shoulder needs to {direction} {abs(diff):.1f} degrees',
            'left_hip': f'Left hip needs to {direction} {abs(diff):.1f} degrees',
            'right_hip': f'Right hip needs to {direction} {abs(diff):.1f} degrees'
        }
        
        return suggestions.get(joint_name, 'Posture needs adjustment')

    def detect_pose_state(self, pose_landmarks):
        if not pose_landmarks:
            return 'No pose detected'
            
        landmarks = pose_landmarks.landmark
        
        # 计算髋关节和膝关节的角度
        left_hip_angle = self.calculate_angle(
            landmarks[23],  # 左髋
            landmarks[25],  # 左膝
            landmarks[27]   # 左踝
        )
        right_hip_angle = self.calculate_angle(
            landmarks[24],  # 右髋
            landmarks[26],  # 右膝
            landmarks[28]   # 右踝
        )
        
        # 计算躯干垂直度（肩部到髋部的垂直比例）
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2  # 肩部中点
        hip_y = (landmarks[23].y + landmarks[24].y) / 2      # 髋部中点
        ankle_y = (landmarks[27].y + landmarks[28].y) / 2    # 踝部中点
        vertical_ratio = abs(shoulder_y - hip_y) / abs(hip_y - ankle_y)
        
        # 判断姿态状态
        hip_angle = np.degrees(min(left_hip_angle, right_hip_angle))
        
        if hip_angle > self.pose_thresholds['standing']['hip_knee_angle'] and \
           vertical_ratio > self.pose_thresholds['standing']['vertical_ratio']:
            return 'Standing'
        elif self.pose_thresholds['sitting']['hip_knee_angle'] < hip_angle < self.pose_thresholds['standing']['hip_knee_angle'] and \
             vertical_ratio > self.pose_thresholds['sitting']['vertical_ratio']:
            return 'Sitting'
        elif vertical_ratio < self.pose_thresholds['sitting']['vertical_ratio']:
            return 'Lying'
        else:
            return 'Other pose'