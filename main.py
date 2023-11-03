from Sequential import Sequential
import numpy as np
from Loader import Loader
import pandas as pd

def create_sequences(data, seq_length):
  sequences = []
  targets = []
  data_len = len(data)
  for i in range(data_len - seq_length):
    seq_end = i + seq_length
    seq_x = data[i:seq_end]
    seq_y = data[seq_end]
    sequences.append(seq_x)
    targets.append(seq_y)
    return np.array(sequences), np.array(targets)
  

loader = Loader()
m2 = loader.load("train.json")
loader.save("model.json", m2)

df = pd.read_csv("train.csv")
df = df.drop(columns=["Date", "Adjusted Close"])

# Define the column you want to normalize
column_to_normalize = 'Volume'

# Calculate the minimum and maximum values of the column
min_value = df[column_to_normalize].min()
max_value = df[column_to_normalize].max()

# Perform Min-Max scaling to normalize the column
df[column_to_normalize] = (df[column_to_normalize] - min_value) / (max_value - min_value)

testX, testY = create_sequences(df.values.tolist(), 4)

print(m2.predict(testX[0]))
print(testY)

m2.summarize()