import json
from datetime import datetime, timedelta
from pathlib import Path

class TrainingPlan:
    def __init__(self):
        self.plans_dir = Path('training_data/plans')
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        
    def create_plan(self, user_info):
        """根据用户信息创建个性化训练计划
        Args:
            user_info: 字典，包含用户的基本信息和训练目标
        Returns:
            训练计划字典
        """
        # 根据用户目标和水平生成训练计划
        plan = {
            'user_id': user_info['user_id'],
            'start_date': datetime.now().strftime('%Y-%m-%d'),
            'duration_weeks': 4,
            'exercises': self._generate_exercises(user_info),
            'progress': {'completed_sessions': 0, 'total_sessions': 0}
        }
        
        # 保存训练计划
        self._save_plan(plan)
        return plan
        
    def _generate_exercises(self, user_info):
        """生成训练动作列表
        Args:
            user_info: 用户信息字典
        Returns:
            训练动作列表
        """
        # 根据用户目标和水平选择适合的动作
        exercise_templates = {
            'beginner': {
                'strength': [
                    {'name': 'squat', 'sets': 3, 'reps': 10, 'target_angles': {
                        'left_hip': 1.57,  # 90度
                        'right_hip': 1.57,
                        'left_knee': 1.57,
                        'right_knee': 1.57
                    }},
                    {'name': 'push_up', 'sets': 3, 'reps': 8, 'target_angles': {
                        'left_shoulder': -0.52,  # -30度
                        'right_shoulder': -0.52,
                        'left_elbow': 1.57,
                        'right_elbow': 1.57
                    }}
                ],
                'flexibility': [
                    {'name': 'standing_stretch', 'duration': 30, 'target_angles': {
                        'left_shoulder': 1.57,
                        'right_shoulder': 1.57
                    }}
                ]
            }
        }
        
        level = user_info.get('level', 'beginner')
        goal = user_info.get('goal', 'strength')
        
        return exercise_templates[level][goal]
        
    def _save_plan(self, plan):
        """保存训练计划到文件
        Args:
            plan: 训练计划字典
        """
        plan_file = self.plans_dir / f"{plan['user_id']}_{plan['start_date']}.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
            
    def load_plan(self, user_id):
        """加载用户的训练计划
        Args:
            user_id: 用户ID
        Returns:
            最新的训练计划字典，如果没有则返回None
        """
        plan_files = list(self.plans_dir.glob(f"{user_id}_*.json"))
        if not plan_files:
            return None
            
        # 获取最新的训练计划
        latest_plan = max(plan_files, key=lambda x: x.stat().st_mtime)
        with open(latest_plan, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def update_progress(self, user_id, session_data):
        """更新训练进度
        Args:
            user_id: 用户ID
            session_data: 训练数据字典
        """
        plan = self.load_plan(user_id)
        if not plan:
            return
            
        plan['progress']['completed_sessions'] += 1
        plan['progress']['last_session'] = session_data
        
        self._save_plan(plan)