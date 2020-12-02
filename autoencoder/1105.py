import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial

tf.reset_default_graph()

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

train_x = train_x.astype(np.float32).reshape(-1,28*28) / 255.0
test_x = test_x.astype()
