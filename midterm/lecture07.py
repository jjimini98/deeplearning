import random
import numpy as np


x0_data = np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]])
b= np.zeros(([1,5]))
wx = random.randint(0,25)
wx = np.reshape(wx,[5,5])
print(np.matmul(x0_data,wx)+b)