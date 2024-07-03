
import pandas as pd
from neuralprophet import NeuralProphet, set_log_level

set_log_level("ERROR")

import pandas as pd

# Load the dataset from the CSV file using pandas
df = pd.read_csv("https://raw.githubusercontent.com/KhawajaAbdullah2000/CSV-files/main/CAR_SALES_DATA_INFLATION.csv")

# Plot the dataset, showing price (y column) over time (ds column)
#plt = df.plot(x="ds", y="y", figsize=(15, 5))

df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
df=df.iloc[:, :2]
df.tail()

df.info()

# Disable logging messages unless there is an error
set_log_level("ERROR")

# Create a NeuralProphet model with default parameters
m = NeuralProphet()


# Fit the model on the dataset (this might take a bit)
metrics = m.fit(df)

# Create a new dataframe reaching 365 into the future for our forecast, n_historic_predictions also shows historic data
df_future = m.make_future_dataframe(df, n_historic_predictions=True, periods=365)

# Predict the future
forecast = m.predict(df_future)



dates= pd.date_range(start="2024-01-01", periods=30, freq='D')
future = pd.DataFrame({
    'ds': dates,
    'y': [None] * len(dates)
})

fore = m.predict(future)
print(fore.head())

import pickle

# Assume 'm' is your trained NeuralProphet model
with open('MyLocalModel.pkl', 'wb') as f:
    pickle.dump(m, f)
    