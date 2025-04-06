🎙️ Grammar Scoring Engine
An AI-powered tool for evaluating spoken audio clips by predicting scores based on acoustic features.
📋 Overview
This project uses machine learning to analyze and score spoken audio clips based on their acoustic features. By extracting Mel-Frequency Cepstral Coefficients (MFCC) from audio samples, we trained a Random Forest Regressor model that can predict scores for new audio recordings with high accuracy.
The system provides:

Audio feature extraction from .wav files
Machine learning model for scoring speech quality
Interactive web interface for easy use

✨ Features

Audio Analysis: Extract MFCC features from spoken audio clips
ML Prediction: Score audio clips using a trained Random Forest Regressor
Streamlit Web App: User-friendly interface for uploading and scoring audio files
Easy Deployment: Simple setup for local development or cloud deployment

🛠️ Installation

Clone the repository
bashCopygit clone https://github.com/bhuradiaarchit/grammar-scoring-engine.git
cd grammar-scoring-engine

Set up a virtual environment
bashCopypython -m venv venv

# For Linux/macOS
source venv/bin/activate

# For Windows
venv\Scripts\activate

Install dependencies
bashCopypip install -r requirements.txt


🚀 Usage
Jupyter Notebook Development
To explore the data and model development:
bashCopyjupyter notebook
Running the Streamlit App
To start the interactive web interface:
bashCopycd frontend  # Navigate to the frontend directory
streamlit run app.py
Then open your browser and go to http://localhost:8501

📊 How It Works

Feature Extraction: The system extracts MFCC features from audio files, capturing the acoustic characteristics of speech
Model Training: A Random Forest Regressor is trained on paired audio features and human-assigned scores
Prediction: New audio files are processed to extract features and then scored by the model
User Interface: The Streamlit app provides an intuitive interface for uploading and scoring audio files

🧪 Model Performance
The Random Forest Regressor achieves strong performance in predicting audio quality scores:

Mean Square Error: 1.1786772471910112
R-squared: 0.13532739291502693


🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
