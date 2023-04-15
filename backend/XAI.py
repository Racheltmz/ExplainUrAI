# Import Libraries
from datetime import datetime

class XAI():
    def __init__(self):
        self.current_epoch = int(datetime.now().timestamp()) # Current datetime in epoch
        self.font = 'Bahnschrift'

    @property
    def get_current_epoch(self):
        return self.current_epoch
    
    @property
    def get_matplotlib_font(self):
        return self.font
