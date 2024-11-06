import pandas as pd
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from preprocess import create_features, cylindrical_encoding
import pickle
import os

class CrowdPredictor:
    def __init__(self):
        # Load model, encoder, and scaler
        self.model = load_model('lstm_model.h5')
        self.encoder = pickle.load(open('binary_encoder.pkl', 'rb'))
        self.scaler = pickle.load(open('min_max_scaler.pkl', 'rb'))
        
        # Load initial data
        self.data = pd.read_csv('synthetic_crowd_data.csv')[-60:]
        self.prev_crowd_count = self.data['crowd_count'].iloc[-1]
        self.data['crowd_count'] = self.data['crowd_count'].shift(1)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data.dropna(inplace=True)
        
        # Initialize current datetime
        self.curr_datetime = self.data['datetime'].iloc[-1] + pd.Timedelta(minutes=1)

    def update_data(self, input_df):
        # Update the internal data attribute
        self.data = pd.concat([self.data, input_df], ignore_index=True)
        self.data = self.data[-60:]  # Keep only the last 60 records
        self.curr_datetime = self.data['datetime'].iloc[-1] + pd.Timedelta(minutes=1)

    def predict_single(self, camera_location):
        os.system('cls')  # Clear console

        # Prepare the input row with updated datetime and previous crowd count
        datetime = self.curr_datetime
        input_df = pd.DataFrame({
            'datetime': [datetime], 
            'camera_location': [camera_location], 
            'crowd_count': [self.prev_crowd_count]
        })
        
        # Update data with new input
        self.update_data(input_df)
        
        # Feature engineering and scaling
        features = create_features(self.data)
        df = cylindrical_encoding(features)
        df = self.encoder.transform(df)
        df[['dayofyear', 'dayofmonth', 'weekofyear']] = self.scaler.transform(df[['dayofyear', 'dayofmonth', 'weekofyear']])

        # Model prediction
        X = np.expand_dims(df, axis=0).astype('float32')
        prediction = self.model.predict(X)
        self.prev_crowd_count = int(prediction[0][0])

        return self.prev_crowd_count

    def predict_batch(self, batch_data):
        os.system('cls')

        df = pd.read_csv(batch_data.name)
        predictions = []

        for index, row in df.iterrows():
            camera_location_input = row['camera_location']
            prediction = self.predict_single(camera_location_input)
            predictions.append(prediction)
        
        # Prepare DataFrame for LinePlot output
        plot_data = self.data[['datetime', 'crowd_count']]

        return ', '.join(map(str, predictions)), plot_data
    
    def predict_nxt_m_minutes(self, camera_location, m):
        os.system('cls')

        predictions = []
        for _ in range(int(m)):  # Ensure m is an integer
            prediction = self.predict_single(camera_location)
            predictions.append(prediction)
        
        # Prepare DataFrame for LinePlot output
        plot_data = self.data[['datetime', 'crowd_count']]

        return ', '.join(map(str, predictions)), plot_data

# Instantiate the predictor
predictor = CrowdPredictor()

# Gradio interface setup
with gr.Blocks() as prediction_block:
    gr.Label("Crowd Count Prediction")
    
    # Single Prediction Tab
    with gr.Tab("Single Prediction"):
        single_camera_location = gr.Dropdown(choices=[f"Camera_{i}" for i in range(1, 101)], label="Camera Location (Category)")
        single_predict_btn = gr.Button("Predict")
        single_result = gr.Number(label="Predicted Crowd Count")

        single_predict_btn.click(
            predictor.predict_single,
            inputs=single_camera_location,
            outputs=single_result
        )

    # Batch Prediction Tab
    with gr.Tab("Batch Prediction"):
        batch_input = gr.File(label="Upload CSV")
        batch_predict_btn = gr.Button("Predict")
        batch_output = gr.Textbox(label="Predicted Crowd Counts")
        batch_plot = gr.LinePlot(
            predictor.data,
            x="datetime", 
            y="crowd_count", 
            title="Crowd Count History", 
            x_title="Datetime", 
            y_title="Crowd Count", 
            label="Crowd Count History"
        )

        batch_predict_btn.click(
            predictor.predict_batch,
            inputs=batch_input,
            outputs=[batch_output, batch_plot]
        )

    # Predict Next M Minutes Tab
    with gr.Tab("Predict next M minutes"):
        m = gr.Number(label="Minutes to predict", minimum=1, maximum=100)
        next_camera_location = gr.Dropdown(choices=[f"Camera_{i}" for i in range(1, 101)], label="Camera Location (Category)")
        predict_next_btn = gr.Button("Predict")
        next_output = gr.Textbox(label="Predicted Crowd Counts")
        next_plot = gr.LinePlot(
            predictor.data,
            x="datetime", 
            y="crowd_count", 
            title="Crowd Count History", 
            x_title="Datetime", 
            y_title="Crowd Count", 
            label="Crowd Count History"
        )

        predict_next_btn.click(
            predictor.predict_nxt_m_minutes,
            inputs=[next_camera_location, m],
            outputs=[next_output, next_plot]
        )

# Launch the Gradio app
prediction_block.launch(share=True, debug=True)
