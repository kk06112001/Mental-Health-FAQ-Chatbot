from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

vectorizer = joblib.load('vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')

data = pd.read_csv("Mental_Health_FAQ.csv")

def get_most_similar_answer(user_input, threshold=0.2):
    user_input_tfidf = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix)
    most_similar_index = cosine_similarities.argmax()
    highest_similarity = cosine_similarities[0][most_similar_index]
    if highest_similarity < threshold:
        return "I'm sorry, I couldn't find a relevant answer. Could you please rephrase your question?"
    return data['Answers'][most_similar_index]

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response = get_most_similar_answer(user_input)
    return response
if __name__ == '__main__':
    app.run(debug=True)
