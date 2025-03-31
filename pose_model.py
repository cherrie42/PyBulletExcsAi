import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PoseNet().to(self.device)
        self.exercise_types = ['squat', 'push_up', 'standing_stretch']
        
    def preprocess_landmarks(self, landmarks):
        """将 MediaPipe 关键点转换为模型输入格式"""
        features = []
        for landmark in landmarks.landmark:
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