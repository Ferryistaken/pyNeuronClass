import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

width = 320
height = 180

img = Image.open('0001.jpg').convert('RGBA') # i should prob use rgb since the change is minimal and it will be a 4 channel data type
resImg = img.resize((width, height), Image.NEAREST)
arr = np.array(resImg)

print(arr)
print(arr.shape)

# record the original shape
shape = arr.shape

# make a 1-dimensional view of arr
flat_arr = arr.ravel()

# convert it to a matrix
vector = np.matrix(flat_arr)

# do something to the vector
# vector[:,::10] = 128

# reform a numpy array of the original shape
arr2 = np.asarray(vector).reshape(shape)

# make a PIL image
# img2 = Image.fromarray(arr2, 'RGBA')
# img2.show()
