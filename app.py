from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  

# Load the model
with open('MyLocalModel.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    periods = int(request.form['periods'])

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        df=df.iloc[:, :2]
        df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
        model.restore_trainer()
        #witout historic data
        df_future = model.make_future_dataframe(df, periods=periods)

        # df_future = model.make_future_dataframe(df, n_historic_predictions=True, periods=periods)
        forecast = model.predict(df_future)
        return jsonify(forecast.to_dict())
    except Exception as e:
        print("error" ,str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/predict_history', methods=['POST'])
def predict_history():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    periods = int(request.form['periods'])

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        df=df.iloc[:, :2]
        df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
        model.restore_trainer()
        #witout historic data
        #df_future = model.make_future_dataframe(df, periods=periods)

        df_future = model.make_future_dataframe(df, n_historic_predictions=True, periods=periods)
        forecast = model.predict(df_future)
        return jsonify(forecast.to_dict())
    except Exception as e:
        print("error" ,str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
