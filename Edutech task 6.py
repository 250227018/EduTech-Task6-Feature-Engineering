import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("C:/Users/hp/OneDrive/Desktop/datasets/Housing.csv")
print(df.head())
print(df.dtypes)

# 2. One-hot encoding (categorical columns ko numbers mein convert karo)
categorical_cols = ['mainroad', 'guestroom', 'basement', 
                    'hotwaterheating', 'airconditioning', 
                    'prefarea', 'furnishingstatus']

df_encoded = pd.get_dummies(df, columns=categorical_cols)
print(df_encoded.head())

# 3. Feature Scaling (numeric columns ko same scale pe lao)
scaler = StandardScaler()
numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

# 4. Interaction Term (naya feature banao)
df_encoded['area_x_bedrooms'] = df_encoded['area'] * df_encoded['bedrooms']

# 5. Save engineered dataset
df_encoded.to_csv('Housing_Engineered.csv', index=False)
print("Done! Engineered dataset saved.")