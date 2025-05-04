import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class DataAnalyzer:
    def __init__(self):
        # 初始化轻量级AI模型 (参数量<14B)
        self.model_name = "microsoft/phi-1_5"  # 1.3B参数的开源模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3  # 反馈分类: 正确/需改进/危险
        ).to(self.device)

    def __init__(self):
        self.data_dir = Path('training_data/sessions')
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_session_data(self, user_id, session_data):
        """保存训练会话数据
        Args:
            user_id: 用户ID
            session_data: 训练会话数据字典
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_file = self.data_dir / f"{user_id}_{timestamp}.json"

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def get_user_sessions(self, user_id, days=30):
        """获取用户最近的训练会话数据
        Args:
            user_id: 用户ID
            days: 查询天数
        Returns:
            训练会话数据列表
        """
        session_files = list(self.data_dir.glob(f"{user_id}_*.json"))
        cutoff_date = datetime.now() - timedelta(days=days)

        sessions = []
        for file in session_files:
            if file.stat().st_mtime >= cutoff_date.timestamp():
                with open(file, 'r', encoding='utf-8') as f:
                    sessions.append(json.load(f))

        return sorted(sessions, key=lambda x: x['timestamp'])

    def analyze_progress(self, user_id):
        """分析用户训练进度
        Args:
            user_id: 用户ID
        Returns:
            包含AI智能反馈的进度分析报告
        """
        sessions = self.get_user_sessions(user_id)
        if not sessions:
            return None

        # 获取最新训练数据
        latest_session = sessions[-1]
        feedback = self._generate_ai_feedback(latest_session)

        # 将AI反馈整合到分析报告中

        # 分析关键指标
        analysis = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'total_sessions': len(sessions),
            'exercise_stats': self._analyze_exercises(sessions),
            'posture_improvement': self._analyze_posture(sessions),
            'consistency_score': self._calculate_consistency(sessions),
            'fitness_level': self._assess_fitness_level(sessions),
            'fatigue_analysis': self._analyze_fatigue(sessions),
            'recommendations': self._generate_recommendations(sessions)
        }

        return analysis

    def _analyze_exercises(self, sessions):
        """分析运动数据统计
        Args:
            sessions: 训练会话数据列表
        Returns:
            运动统计数据字典
        """
        exercise_stats = {}
        for session in sessions:
            for exercise in session.get('exercises', []):
                name = exercise['name']
                if name not in exercise_stats:
                    exercise_stats[name] = {
                        'total_count': 0,
                        'total_duration': 0,
                        'accuracy_scores': []
                    }

                stats = exercise_stats[name]
                stats['total_count'] += 1
                stats['total_duration'] += exercise.get('duration', 0)
                if 'accuracy_score' in exercise:
                    stats['accuracy_scores'].append(exercise['accuracy_score'])

        # 计算平均值和进步趋势
        for name, stats in exercise_stats.items():
            if stats['accuracy_scores']:
                stats['avg_accuracy'] = np.mean(stats['accuracy_scores'])
                stats['accuracy_trend'] = self._calculate_trend(
                    stats['accuracy_scores'])

        return exercise_stats

    def _analyze_posture(self, sessions):
        """分析姿态改进情况
        Args:
            sessions: 训练会话数据列表
        Returns:
            姿态分析数据字典
        """
        joint_improvements = {}
        for session in sessions:
            for joint, data in session.get('joint_accuracy', {}).items():
                if joint not in joint_improvements:
                    joint_improvements[joint] = []
                joint_improvements[joint].append(data['accuracy'])

        # 计算每个关节的改进趋势
        improvements = {}
        for joint, scores in joint_improvements.items():
            improvements[joint] = {
                'current': scores[-1] if scores else 0,
                'trend': self._calculate_trend(scores)
            }

        return improvements

    def _calculate_trend(self, values):
        """计算数据趋势
        Args:
            values: 数值列表
        Returns:
            趋势值（正值表示上升趋势，负值表示下降趋势）
        """
        if len(values) < 2:
            return 0

        x = np.arange(len(values))
        y = np.array(values)
        z = np.polyfit(x, y, 1)
        return z[0]  # 返回斜率

    def _calculate_consistency(self, sessions):
        """计算训练一致性得分
        Args:
            sessions: 训练会话数据列表
        Returns:
            一致性得分（0-100）
        """
        if len(sessions) < 2:
            return 100 if sessions else 0

        # 计算会话间隔
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in sessions]
        intervals = [(t2 - t1).days for t1,
                     t2 in zip(timestamps[:-1], timestamps[1:])]

        # 根据间隔计算一致性得分
        avg_interval = np.mean(intervals)
        consistency = 100 * np.exp(-0.1 * avg_interval)  # 间隔越大，得分越低

        return round(consistency, 2)

    def generate_progress_charts(self, user_id, save_dir=None):
        """生成进度图表
        Args:
            user_id: 用户ID
            save_dir: 保存目录路径，如果为None则显示图表
        """
        sessions = self.get_user_sessions(user_id)
        if not sessions:
            return

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # 绘制运动准确度趋势
        self._plot_accuracy_trend(sessions, ax1)

        # 绘制训练频率统计
        self._plot_frequency_stats(sessions, ax2)

        plt.tight_layout()

        if save_dir:
            save_path = Path(
                save_dir) / f"{user_id}_progress_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def _generate_ai_feedback(self, session_data):
        """使用本地AI模型生成训练反馈
        Args:
            session_data: 单次训练会话数据
        Returns:
            包含AI反馈的字典
        """
        # 准备输入数据
        input_text = self._prepare_feedback_input(session_data)
        inputs = self.tokenizer(
            input_text, return_tensors="pt").to(self.device)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

        # 生成反馈
        feedback_types = ["姿势正确", "需要改进", "危险动作"]
        feedback = {
            "overall": feedback_types[predictions[0].item()],
            "details": self._generate_detailed_feedback(session_data)
        }
        return feedback

    def _prepare_feedback_input(self, session_data):
        """准备模型输入文本"""
        # 将训练数据转换为自然语言描述
        exercises = ", ".join([ex['name']
                              for ex in session_data.get('exercises', [])])
        accuracy_stats = ", ".join(
            f"{joint}:{data['accuracy']}%"
            for joint, data in session_data.get('joint_accuracy', {}).items()
        )
        return f"训练动作: {exercises}. 关节准确率: {accuracy_stats}."

    def _generate_detailed_feedback(self, session_data):
        """生成详细改进建议"""
        details = []
        for joint, data in session_data.get('joint_accuracy', {}).items():
            if data['accuracy'] < 70:
                details.append(f"{joint}关节准确率较低({data['accuracy']}%)，建议加强训练")
        return details

    def _plot_accuracy_trend(self, sessions, ax):
        """绘制准确度趋势图
        Args:
            sessions: 训练会话数据列表
            ax: matplotlib轴对象
        """
        dates = [datetime.fromisoformat(s['timestamp']) for s in sessions]
        accuracies = []

        for session in sessions:
            avg_accuracy = np.mean(
                [j['accuracy'] for j in session.get('joint_accuracy', {}).values()])
            accuracies.append(avg_accuracy)

        ax.plot(dates, accuracies, marker='o')
        ax.set_title('训练准确度趋势')
        ax.set_xlabel('日期')
        ax.set_ylabel('平均准确度 (%)')
        ax.grid(True)

    def _plot_frequency_stats(self, sessions, ax):
        """绘制训练频率统计图
        Args:
            sessions: 训练会话数据列表
            ax: matplotlib轴对象
        """
        # 统计每周训练次数
        weekly_counts = {}
        for session in sessions:
            date = datetime.fromisoformat(session['timestamp'])
            week = date.strftime('%Y-%W')
            weekly_counts[week] = weekly_counts.get(week, 0) + 1

        weeks = list(weekly_counts.keys())
        counts = list(weekly_counts.values())

        ax.bar(weeks, counts)
        ax.set_title('每周训练频率')
        ax.set_xlabel('周')
        ax.set_ylabel('训练次数')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
