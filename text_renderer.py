from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

class TextRenderer:
    def __init__(self):
        # Get system font path
        self.font_path = os.path.join(os.environ['SystemRoot'], 'Fonts', 'arial.ttf')
        if not os.path.exists(self.font_path):
            # If Arial is not found, try using Calibri
            self.font_path = os.path.join(os.environ['SystemRoot'], 'Fonts', 'calibri.ttf')
    
    def put_text(self, img, text, position, font_size=32, color=(255, 255, 255)):
        """Draw text on image
        Args:
            img: OpenCV image (BGR format)
            text: Text to draw
            position: Text position, tuple (x, y)
            font_size: Font size
            color: Font color, tuple (B, G, R)
        Returns:
            Image with added text
        """
        # Convert OpenCV image to PIL image
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Create drawing object
        draw = ImageDraw.Draw(img_pil)
        
        # Load font
        font = ImageFont.truetype(self.font_path, font_size)
        
        # Draw text
        draw.text(position, text, font=font, fill=color[::-1])  # PIL uses RGB order
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)