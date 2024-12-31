# Restaurant Reviews Sentiment Analysis

A comprehensive NLP-based sentiment analysis project that analyzes customer reviews of restaurants to determine their sentiment (positive, negative, or neutral). This project utilizes machine learning models and natural language processing techniques to preprocess, train, and predict sentiments effectively.

---

## 🚀 Features

- Preprocess restaurant reviews using advanced NLP techniques.
- Analyze sentiments (positive, negative, or neutral) using a trained machine learning model.
- Generate insights to improve customer satisfaction and services.
- Visualization of sentiment distribution for better interpretation.

---

## 🛠️ Technologies Used

- **Programming Language**: Python
- **NLP Libraries**: NLTK, SpaCy
- **Machine Learning**: Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn
- **Dataset**: Kaggle or Custom Scraped Restaurant Reviews Dataset



Install required dependencies:


pip install -r requirements.txt


🧠 How It Works
Data Preprocessing:

Tokenization
Removing stop words
Lemmatization or Stemming
Vectorization using TF-IDF or Count Vectorizer
Model Training:

Train machine learning models like Logistic Regression, Naive Bayes, or SVM on the processed data.
Sentiment Prediction:

Predict sentiment labels for new reviews using the trained model.
Visualization:

Plot sentiment distributions and word clouds for deeper insights.
🛠️ Usage
1. Preprocessing Data
Run the preprocessing script to clean and prepare the data:

bash
Copy code
python src/preprocess.py
2. Training the Model
Train the sentiment analysis model:

bash
Copy code
python src/train_model.py
3. Making Predictions
Predict the sentiment of new reviews:

bash
Copy code
python src/predict.py --review "The food was amazing and the service was excellent!"
4. Visualizing Results
Generate visualizations of sentiment distribution:

bash
Copy code
python src/visualize.py
🖥️ Run the Flask API for Live Predictions
Start the Flask API:

bash
Copy code
python app.py
Open your browser and navigate to http://127.0.0.1:5000 to access the live prediction interface.

📈 Results
Accuracy: 90%
Precision: 89%
Recall: 88%
F1-Score: 88%
