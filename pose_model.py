import torch
import torch.nn as nn
import torch.nn.functional as F

# PoseNet Model Definition
# Parameter Count Calculation:
# fc1: 99 * 128 + 128 = 12800
# fc2: 128 * 64 + 64 = 8256
# fc3: 64 * 3 + 3 = 195
# Total: 12800 + 8256 + 195 = 21251 parameters
# This is significantly less than the 14 billion parameter limit.


class PoseNet(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_classes=3):
        super(PoseNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class PosePrediction:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PoseNet().to(self.device)
        self.exercise_types = ['squat', 'push_up', 'standing_stretch']

    def preprocess_landmarks(self, landmarks):
        """将 MediaPipe 关键点转换为模型输入格式"""
        features = []
        if hasattr(landmarks, 'landmark'):
            # 处理MediaPipe PoseLandmarkerResult格式
            for landmark in landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            # 处理直接传递的landmark列表格式
            for landmark in landmarks:
                features.extend([landmark.x, landmark.y, landmark.z])
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def predict(self, landmarks):
        """预测姿势类型"""
        self.model.eval()
        with torch.no_grad():
            features = self.preprocess_landmarks(landmarks)
            outputs = self.model(features)
            _, predicted = torch.max(outputs, 1)
            return self.exercise_types[predicted.item()]
