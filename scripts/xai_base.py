# Import Libraries
import os
from datetime import datetime

class XAI():
    def __init__(self):
        self.report_dir = '../reports'
        self.report_convention = f'{self.report_dir}/explainurai_'
        self.current_epoch = int(datetime.now().timestamp()) # Current datetime in epoch
        self.font = 'Bahnschrift'
        self.create_dir_if_notexists()

    @property
    def get_report_convention(self):
        return self.report_convention

    @property
    def get_current_epoch(self):
        return self.current_epoch
    
    @property
    def get_matplotlib_font(self):
        return self.font

    def create_dir_if_notexists(self):
        os.mkdir(self.report_dir)