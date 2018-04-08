import pandas as pd
import numpy as np
import time

start = time.time()
df = pd.read_csv("../../Data/datafestsmall.csv", sep=',')
df_actually_small = df.sample(n = 1000)
df_actually_small.to_csv("../Data/datafest_playground.csv", sep=',')

end = time.time()

print ("Took %fs to open") % (end - start,)


