import numpy as np 
from PIL import Image
import csv
from SDC_CNN import SDC_CNN

# Load the data
data = []
true_values = []
with open('../training/driving_log.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
	    I = np.asarray(Image.open(row[0]))
	    data.append(I)
	    true_values.append(row[3])

data = np.asarray(data)
true_values = np.asarray(true_values)
print("data loaded")

input_dim = list(data[1,:,:,:].shape)

# Create the convolutional neural network
sdc_cnn = SDC_CNN(input_dim)
sdc_cnn.train(data, true_values)





