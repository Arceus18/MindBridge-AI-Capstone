import pandas as pd

# Try to read the file
try:
    df = pd.read_csv("data/counseling.csv")
    print("✅ File loaded successfully!")
    print("👇 HERE ARE YOUR EXACT COLUMN NAMES:")
    print(df.columns.tolist())
except Exception as e:
    print(f"❌ Could not read file: {e}")