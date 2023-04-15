# Import Libraries
from io import BytesIO
import plotly.io as pio
from reportlab.platypus import Flowable
import reportlab

class PlotlyGraph(Flowable):
    def __init__(self, fig, width=400, height=300):
        self.fig = fig
        self.width = width
        self.height = height
    # Troubleshoot
    def draw(self):
        img = BytesIO(pio.to_image(self.fig, format='png'))
        img_ratio = self.height / float(self.width)
        self.canv.drawImage(img, 0, 0, self.width, self.height * img_ratio)

    def wrap(self, availWidth, availHeight):
        return self.width, self.height