import numpy as np


#numpy linspace 함수
lis = [0,1,2,3,4,5]
space = np.linspace(0,100,5)
s_space = np.linspace(lis,1)
# print(space)
# print(type(space))
# print(s_space)

# sin 함수

t_min , t_max = 0, 30
resolution = 0.1

t = np.linspace(t_min, t_max ,int((t_max- t_min)/ resolution))


