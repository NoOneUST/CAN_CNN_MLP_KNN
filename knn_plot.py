import matplotlib.pyplot as plt
import numpy as np
x_axis = range(1, 11)
accuRes=[0.9615999999999104,
         0.9615999999999104,
         0.9634999999999102,
         0.9638999999999102,
         0.9619999999999104,
         0.9631999999999102,
         0.9597999999999106,
         0.9607999999999105,
         0.9598999999999106,
         0.9610999999999105]
print(np.ones([1,10])-accuRes)
plt.figure()
# plt.plot(x_axis, np.ones([10,])-accuRes)
plt.plot(x_axis, accuRes)
plt.show()