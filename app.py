import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from flask import Flask, render_template, request, send_file
import pandas as pd
import io
import os
import logging
import csv
import chardet
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import tempfile

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO, filename='/tmp/app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

# Utility Functions
def detect_encoding(file):
    raw_data = file.read(2048)
    result = chardet.detect(raw_data)
    file.seek(0)
    return result['encoding'] or 'utf-8'

def detect_delimiter(file, encoding='utf-8'):
    file.seek(0)
    sample = file.read(2048).decode(encoding, errors='replace')
    sniffer = csv.Sniffer()
    try:
        return sniffer.sniff(sample).delimiter
    except csv.Error:
        logging.warning("Could not detect delimiter, defaulting to comma")
        return ','
    finally:
        file.seek(0)

def load_file(uploaded_file):
    logging.info(f"Loading file: {uploaded_file.filename}")
    try:
        filename = uploaded_file.filename.lower()
        encoding = detect_encoding(uploaded_file)

        if filename.endswith(('.csv', '.txt', '.tsv')):
            delimiter = detect_delimiter(uploaded_file, encoding)
            logging.info(f"Detected delimiter: '{delimiter}'")
            df = pd.read_csv(uploaded_file, encoding=encoding, sep=delimiter, on_bad_lines='warn')
        else:
            logging.error(f"Unsupported file format: {filename}")
            return None

        if df.empty:
            logging.warning(f"DataFrame is empty for {uploaded_file.filename}")
            return None
        logging.info(f"Loaded DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading file {uploaded_file.filename}: {e}")
        return None

def analyze_emotion(text):
    if not isinstance(text, str) or not text.strip():
        return "Neutral"
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode('utf8')

def generate_emotion_plot(df, text_column):
    df['emotion'] = df[text_column].apply(analyze_emotion)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='emotion', data=df, palette='coolwarm', order=['Positive', 'Neutral', 'Negative'])
    ax.set_title("Emotion Distribution")
    return plot_to_base64(fig)

# Flask Routes
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering index.html: {str(e)}")
        return "Server Error: Template issue", 500

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '').strip()
    if not text and 'file' not in request.files:
        return render_template('index.html', error="Please enter text or upload a file.")

    if 'file' in request.files:
        file = request.files['file']
        if file.filename:
            df = load_file(file)
            if df is None:
                return render_template('index.html', error="Failed to load file. Check format or see app.log.")
            if df.empty:
                return render_template('index.html', error="File is empty or could not be parsed.")

            text_candidates = [col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower()]
            text_column = text_candidates[0] if text_candidates else df.columns[0]

            df['emotion'] = df[text_column].apply(analyze_emotion)
            plot = generate_emotion_plot(df, text_column)

            temp_file_path = '/tmp/processed_data.csv'
            df.to_csv(temp_file_path, index=False)

            return render_template('data_preview.html',
                                   data=df.head().to_html(classes='table'),
                                   text_column=text_column,
                                   plot=plot,
                                   columns=df.columns.tolist())

    emotion = analyze_emotion(text)
    return render_template('data_preview.html',
                           prediction=emotion,
                           input_text=text)

@app.route('/download')
def download():
    temp_file_path = '/tmp/processed_data.csv'
    if not os.path.exists(temp_file_path):
        return render_template('data_preview.html', error="No data to download.")
    
    return send_file(temp_file_path, mimetype='text/csv', as_attachment=True, download_name='processed_data.csv')

if __name__ == '__main__':
    app.run(debug=True)
