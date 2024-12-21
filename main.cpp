import pandas as pd
train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
data_dictionary = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/data_dictionary.csv')

display(train.head())
print(f"Train shape: {train.shape}")

display(test.head())
print(f"Test shape: {test.shape}")

display(data_dictionary.head())
print(f"Dictionary shape: {data_dictionary.shape}")
