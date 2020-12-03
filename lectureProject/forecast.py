import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np

selected_cols = np.delete(np.arrange(0,19),17)
xy_train = np.loadtxt('./dataset_6.csv', delimiter=',', skiprows = 1)