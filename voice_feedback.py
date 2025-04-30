import pyttsx3
from datetime import datetime
import threading


class VoiceFeedback:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        self.last_voice_time = datetime.now()
        self.last_pose_state = None
        self.last_evaluation = None

    def speak(self, message, priority=False):
        """语音播报反馈信息
        Args:
            message: 要播报的信息
            priority: 是否为优先消息（优先消息不受时间间隔限制）
        """
        current_time = datetime.now()
        if priority or (current_time - self.last_voice_time).seconds >= 3:
            def speak_thread():
                self.engine.say(message)
                self.engine.runAndWait()
            t = threading.Thread(target=speak_thread)
            t.daemon = True
            t.start()
            self.last_voice_time = current_time

    def update_pose_state(self, pose_state):
        """更新并播报姿势状态"""
        if pose_state != self.last_pose_state:
            feedback_messages = {
                'Standing': '站立姿势',
                'Squat': '深蹲姿势',
                'Lying': '俯卧姿势',
                'Invalid': '请调整姿势',
                'Preparing': '准备开始'
            }
            if pose_state in feedback_messages:
                self.speak(feedback_messages[pose_state])
            self.last_pose_state = pose_state

    def report_exercise_completion(self, exercise_name, rep_count, total_reps=None):
        """报告运动完成情况"""
        exercise_messages = {
            'squat': {
                'progress': f'完成第{rep_count}个深蹲',
                'complete': '深蹲训练完成'
            },
            'push_up': {
                'progress': f'完成第{rep_count}个俯卧撑',
                'complete': '俯卧撑训练完成'
            }
        }

        if exercise_name in exercise_messages:
            if total_reps and rep_count >= total_reps:
                self.speak(
                    exercise_messages[exercise_name]['complete'], priority=True)
            else:
                self.speak(exercise_messages[exercise_name]['progress'])

    def provide_posture_feedback(self, evaluation):
        """提供姿势调整建议"""
        if evaluation != self.last_evaluation:
            feedback = []
            for joint, result in evaluation.items():
                if result['status'] != 'good':
                    joint_names = {
                        'left_shoulder': '左肩',
                        'right_shoulder': '右肩',
                        'left_hip': '左髋',
                        'right_hip': '右髋',
                        'left_knee': '左膝',
                        'right_knee': '右膝'
                    }
                    joint_name = joint_names.get(joint, joint)
                    feedback.append(f"{joint_name}{result['suggestion']}")

            if feedback:
                self.speak("请注意：" + "，".join(feedback))
            self.last_evaluation = evaluation

    def announce_session_status(self, status, duration=None):
        """播报训练会话状态"""
        status_messages = {
            'start': '训练开始，请做好准备',
            'pause': '训练暂停',
            'resume': '训练继续',
            'complete': '训练完成，做得很好！',
            # 'duration' 状态单独处理
        }
        if status in status_messages:
            self.speak(status_messages[status], priority=True)
        elif status == 'duration' and duration is not None:
            try:
                duration_message = f'已训练{int(duration)//60}分钟{int(duration)%60}秒'
                self.speak(duration_message, priority=True)
            except TypeError:
                print(f"Error formatting duration: {duration}")  # 添加错误处理
        # 如果 status 是 'duration' 但 duration 是 None，则不播报
