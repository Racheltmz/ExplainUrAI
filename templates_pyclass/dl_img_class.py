# Import class
import sys
sys.path.insert(0, '../scripts/')
from deep_learning_image import DLImage

# Inputs
model = '../models/cifar10.h5'
imgs = ['../images/cifar.png']

# Generate report
dl_img_xai = DLImage(model, imgs)
dl_img_xai.generate_report()