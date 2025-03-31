import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pose_model import PoseNet
import numpy as np
import json
from pathlib import Path
import mediapipe as mp
import cv2
import pybullet as p

class PoseDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.labels = []
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.load_data(data_path)
        
    def load_data(self, data_path):
        data_dir = Path(data_path)
        exercise_types = {'squat': 0, 'push_up': 1, 'standing_stretch': 2}
        
        for exercise_type in exercise_types:
            exercise_path = data_dir / exercise_type
            if not exercise_path.exists():
                continue
                
            for img_path in exercise_path.glob('*.jpg'):
                # 读取图像并提取关键点
                image = cv2.imread(str(img_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)
                
                if results.pose_landmarks:
                    # 提取关键点特征
                    features = []
                    for landmark in results.pose_landmarks.landmark:
                        features.extend([landmark.x, landmark.y, landmark.z])
                    
                    self.data.append(torch.FloatTensor(features))
                    self.labels.append(exercise_types[exercise_type])
        
        self.data = torch.stack(self.data)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def calculate_pose_error(current_pose, target_pose):
    """计算当前姿势与标准姿势的误差"""
    error = 0
    for curr, target in zip(current_pose, target_pose):
        error += np.sqrt(np.sum((curr - target) ** 2))
    return error

def train_model(data_path, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseNet().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    dataset = PoseDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'models/best_pose_model.pth')

if __name__ == '__main__':
    # 训练模型
    train_model('training_data/pose_images')