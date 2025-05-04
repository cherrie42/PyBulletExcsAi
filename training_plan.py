import json
from datetime import datetime, timedelta
from pathlib import Path


class TrainingPlan:
    def __init__(self):
        self.plans_dir = Path('training_data/plans')
        self.plans_dir.mkdir(parents=True, exist_ok=True)

    def create_plan(self, user_info, analysis_data=None):
        """根据用户信息和分析数据创建个性化训练计划
        Args:
            user_info: 字典，包含用户的基本信息和训练目标
            analysis_data: 可选，来自 DataAnalyzer 的分析结果
        Returns:
            训练计划字典
        """
        # 根据用户目标、水平和分析数据生成训练计划
        plan = {
            'user_id': user_info['user_id'],
            'start_date': datetime.now().strftime('%Y-%m-%d'),
            'duration_weeks': 4,
            'exercises': self._generate_exercises(user_info, analysis_data),
            'progress': {'completed_sessions': 0, 'total_sessions': 0},
            'adaptive_settings': {
                'intensity_adjustment': 0.0,
                'frequency_adjustment': 1.0,
                'difficulty_level': user_info.get('level', 'beginner')
            }
        }

        # 根据分析数据动态调整计划
        if analysis_data:
            # 根据疲劳分析调整训练强度
            if 'fatigue_analysis' in analysis_data:
                fatigue_score = analysis_data['fatigue_analysis'].get(
                    'score', 0)
                plan['adaptive_settings']['intensity_adjustment'] = max(
                    -0.3, min(0.3, -fatigue_score * 0.1))

            # 根据一致性分数调整训练频率
            if 'consistency_score' in analysis_data:
                consistency = analysis_data['consistency_score'] / 100.0
                plan['adaptive_settings']['frequency_adjustment'] = min(
                    1.5, max(0.7, consistency))

            # 根据体能水平调整难度
            if 'fitness_level' in analysis_data:
                plan['adaptive_settings']['difficulty_level'] = analysis_data['fitness_level']

        # 保存训练计划
        self._save_plan(plan)
        return plan

    def _generate_exercises(self, user_info, analysis_data=None):
        """生成训练动作列表，并根据分析数据调整
        Args:
            user_info: 用户信息字典
            analysis_data: 可选，来自 DataAnalyzer 的分析结果
        Returns:
            训练动作列表
        """
        # 基础训练模板
        exercise_templates = {
            'beginner': {
                'strength': [
                    {'name': 'squat', 'sets': 3, 'reps': 10, 'target_angles': {
                        'left_hip': 1.57, 'right_hip': 1.57,
                        'left_knee': 1.57, 'right_knee': 1.57
                    }},
                    {'name': 'push_up', 'sets': 3, 'reps': 8, 'target_angles': {
                        'left_shoulder': -0.52, 'right_shoulder': -0.52,
                        'left_elbow': 1.57, 'right_elbow': 1.57
                    }}
                ],
                'flexibility': [
                    {'name': 'standing_stretch', 'sets': 1, 'duration': 30, 'target_angles': {
                        'left_shoulder': 1.57, 'right_shoulder': 1.57
                    }}
                ]
            },
            'intermediate': {
                # ... (可以添加中级模板)
            },
            'advanced': {
                # ... (可以添加高级模板)
            }
        }

        level = user_info.get('level', 'beginner')
        goal = user_info.get('goal', 'strength')

        # 获取基础练习列表
        exercises = exercise_templates.get(level, {}).get(goal, []).copy()

        # 如果有分析数据，则动态调整计划
        if analysis_data and 'exercise_stats' in analysis_data:
            for exercise in exercises:
                ex_name = exercise['name']
                if ex_name in analysis_data['exercise_stats']:
                    stats = analysis_data['exercise_stats'][ex_name]
                    # 示例：如果准确度趋势良好，增加次数或组数
                    if stats.get('accuracy_trend', 0) > 0.1:  # 假设趋势大于0.1表示进步明显
                        if 'reps' in exercise:
                            exercise['reps'] = min(
                                exercise['reps'] + 2, 20)  # 增加次数，但不超过上限
                        elif 'duration' in exercise:
                            exercise['duration'] = min(
                                exercise['duration'] + 10, 60)  # 增加时长
                    # 示例：如果准确度较低，减少次数或组数
                    elif stats.get('avg_accuracy', 100) < 70:  # 假设平均准确度低于70%
                        if 'reps' in exercise:
                            exercise['reps'] = max(
                                exercise['reps'] - 2, 5)  # 减少次数，但不低于下限
                        elif 'duration' in exercise:
                            exercise['duration'] = max(
                                exercise['duration'] - 5, 15)  # 减少时长

                    # 示例：根据一致性调整组数 (如果一致性好，可以适当增加挑战)
                    if analysis_data.get('consistency_score', 0) > 80:  # 假设一致性得分大于80
                        if 'sets' in exercise:
                            exercise['sets'] = min(exercise['sets'] + 1, 5)
                    elif analysis_data.get('consistency_score', 100) < 50:  # 一致性较差
                        if 'sets' in exercise:
                            exercise['sets'] = max(exercise['sets'] - 1, 1)

        return exercises

    def _save_plan(self, plan):
        """保存训练计划到文件
        Args:
            plan: 训练计划字典
        """
        plan_file = self.plans_dir / \
            f"{plan['user_id']}_{plan['start_date']}.json"
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

        # 更新完成的训练组数
        plan['progress']['completed_sessions'] += 1

        # 更新总训练组数（根据训练计划中的sets设置）
        if plan['exercises']:
            plan['progress']['total_sessions'] = sum(
                ex.get('sets', 1) for ex in plan['exercises']  # 使用get避免缺少sets键
            )

        plan['progress']['last_session_summary'] = {
            'timestamp': session_data['timestamp'],
            # 假设session_data有总时长
            'total_duration': session_data.get('total_duration', 0),
            'avg_accuracy': np.mean([j['accuracy'] for j in session_data.get('joint_accuracy', {}).values()]) if session_data.get('joint_accuracy') else 0
        }

        self._save_plan(plan)
