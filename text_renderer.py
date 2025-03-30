from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

class TextRenderer:
    def __init__(self):
        # 获取系统字体路径
        self.font_path = os.path.join(os.environ['SystemRoot'], 'Fonts', 'msyh.ttc')
        if not os.path.exists(self.font_path):
            # 如果找不到微软雅黑，尝试使用宋体
            self.font_path = os.path.join(os.environ['SystemRoot'], 'Fonts', 'simsun.ttc')
    
    def put_text(self, img, text, position, font_size=32, color=(255, 255, 255)):
        """在图片上绘制中文文本
        Args:
            img: OpenCV图像(BGR格式)
            text: 要绘制的文本
            position: 文本位置，元组(x, y)
            font_size: 字体大小
            color: 字体颜色，元组(B, G, R)
        Returns:
            添加文本后的图像
        """
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 创建绘图对象
        draw = ImageDraw.Draw(img_pil)
        
        # 加载字体
        font = ImageFont.truetype(self.font_path, font_size)
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color[::-1])  # PIL使用RGB顺序
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)