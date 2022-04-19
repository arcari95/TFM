import random
import numpy as np

# (a,b,c,d) = (1,2,3,4) if (random.randint(0,100)>50) else (4,3,2,1)
#
# print(a,b,c,d)
random.seed(10000)
print(random.uniform(0.7, 3))
np.random.seed(10)
print(np.random.randint(0,100))
