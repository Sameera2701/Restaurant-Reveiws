from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('Restaurant_review_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    input_vector = vectorizer.transform([review])
    prediction = model.predict(input_vector)[0]
    sentiment = "Thank You for visiting." if prediction == 1 else "Thank You for your valuable feedback we will improve."
    return render_template('result.html', review=review, prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
