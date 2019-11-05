import matplotlib.pyplot as plt
import numpy as np
# x_axis = range(1, 11)
x_axis = [1,2,4,8,16,32]
accuRes = [0.2321,0.443,0.4434,0.7863,0.9369,0.9629]
# print(np.ones([1,10])-accuRes)
plt.figure()
# plt.plot(x_axis, np.ones([10,])-accuRes)
plt.plot(x_axis, accuRes)
plt.show()