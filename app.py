from flask import Flask, request, jsonify
import pandas as pd
import pickle
from neuralprophet import NeuralProphet

app = Flask(__name__)
#C:\Users\ab\Desktop\LuckyMotosApi\LuckyMotorsmodel.pkl

# Load the model
with open('LuckyMotorsmodel2.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get date range from the request
    data = request.get_json(force=True)
    dates = pd.date_range(start=data['start_date'], periods=data['periods'], freq='D')
    
    df = pd.read_csv("https://raw.githubusercontent.com/KhawajaAbdullah2000/CSV-files/main/CAR_SALES_DATA_INFLATION.csv")
    df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
    df=df.iloc[:, :2]
    #df_future = model.make_future_dataframe(df, n_historic_predictions=True, periods=365)


    
    future = pd.DataFrame({
    'ds': dates,
    'y': [None] * len(dates)
})
    
    # Use the model to predict the future
    try:
        model.restore_trainer()
        forecast = model.predict(future)
        return jsonify(forecast.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

