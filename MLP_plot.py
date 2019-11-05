import matplotlib.pyplot as plt
import numpy as np
# x_axis = range(1, 11)
x_axis = [4, 8, 16, 32, 64, 128, 256]
accuRes = [0.6331, 0.7661, 0.8849, 0.8963, 0.9279, 0.9414, 0.9566]
# print(np.ones([1,10])-accuRes)
plt.figure()
# plt.plot(x_axis, np.ones([10,])-accuRes)
plt.plot(x_axis, accuRes)
plt.show()