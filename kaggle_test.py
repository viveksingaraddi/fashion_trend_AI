import kagglehub
import pandas as pd

# 1. Place your absolute file path inside the quotes
file_path = "../fashion-trend-detection/eCommerce-dataset.csv"

# 2. Load the file from your computer
df = pd.read_csv(file_path)

# 3. Print rows 5 to 10
print(df.iloc[4:10])



