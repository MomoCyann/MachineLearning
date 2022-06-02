import numpy as np
import pandas as pd
a = np.array([[1, 4,5,2],
       [5, 6,5,1]])
s1 = 100
s2 = 75
s3 = 50
s4 = 25
score_weight = np.array([s1, s2, s3, s4])
s = pd.DataFrame({'score':np.multiply(a, score_weight).sum(1)})
print(s)

print(first.join(s, lsuffix='_left', rsuffix='_right'))