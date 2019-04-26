import numpy as np 
from PIL import Image
from SDC_CNN import SDC_CNN
import scipy

file = "center_2019_04_23_19_59_46_854.jpg"
file = "../training_2/IMG/{0}".format(file)
image = np.asarray(Image.open(file))
image_resized = scipy.misc.imresize(image, [66, 200])
sdc_cnn = SDC_CNN([66,200,3])
steer = sdc_cnn.predict(image_resized)

print("steer : {0}".format(steer*25))