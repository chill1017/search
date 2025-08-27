import pandas as pd
import matplotlib.pyplot as plt

file_path = '/Users/calebhill/Documents/misc_coding/search/state_data_3.csv'

df = pd.read_csv(file_path)

plt.hist(df['two_norm'], bins=50, label='two')
plt.hist(df['inv_norm'], bins=50, label='inv')
plt.hist(df['taxi_norm'], bins=50, label='taxi')
plt.legend()
plt.show()