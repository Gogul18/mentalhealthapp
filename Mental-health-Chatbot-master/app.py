import json
import nltk
import mysql.connector
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import random
from flask import Flask, render_template, request, jsonify  # Added jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the Flask app
app = Flask(__name__)
app.static_folder = 'static'

# Initialize NLTK
nltk.download('popular')
lemmatizer = WordNetLemmatizer()

# Load the pre-trained model and other data
model = load_model(r'model.h5')
intents = json.loads(open(r'intents.json').read())
words = pickle.load(open(r'texts.pkl', 'rb'))
classes = pickle.load(open(r'labels.pkl', 'rb'))

# Set up a MySQL database connection
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Gogul@2003",
    database="gogul"
)

cursor = db_connection.cursor()

# Create the chat_history table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_input TEXT,
                    bot_response TEXT,
                    user_sentiment VARCHAR(255),  # Added column for user sentiment
                    bot_sentiment VARCHAR(255)    # Added column for bot sentiment
                )''')
db_connection.commit()

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into an array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create a short form for the word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if the current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by the strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def analyze_sentiment_vader(text):
    sentiment = analyzer.polarity_scores(text)
    compound_score = sentiment['compound']
    
    if compound_score >= 0.6:
        return "very_positive"
    
    elif 0.3 <= compound_score < 0.6:
        return "positive"
    elif 0.1 <= compound_score < 0.3:
        return "somewhat_positive"
    elif -0.1 <= compound_score < 0.1:
        return "neutral"
    elif -0.3 <= compound_score < -0.1:
        return "somewhat_negative"
    elif -0.6 <= compound_score < -0.3:
        return "negative"
    elif compound_score <= -0.6:
        return "very_negative"
    elif compound_score >= 0.5:
        return "joy"
    elif compound_score <= -0.5:
        return "sadness"
    elif compound_score >= 0.4:
        return "surprise"
    elif compound_score <= -0.4:
        return "disgust"
    elif compound_score >= 0.3:
        return "anticipation"
    elif compound_score <= -0.3:
        return "fear"
    elif compound_score >= 0.2:
        return "trust"
    elif compound_score <= -0.2:
        return "anger"
    elif compound_score >= 0.1:
        return "love"
    elif compound_score <= -0.1:
        return "hatred"
    elif compound_score >= 0.05:
        return "gratitude"
    elif compound_score <= -0.05:
        return "regret"
    else:
        return "casual"

def chatbot_response(msg):
    # Analyze sentiment of user input
    user_sentiment = analyze_sentiment_vader(msg)
    
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    
    # Analyze sentiment using VADER
    bot_sentiment = analyze_sentiment_vader(res)
    
    # Save the user input, bot's response, and sentiments to the database
    cursor.execute('INSERT INTO chat_history (user_input, bot_response, user_sentiment, bot_sentiment) VALUES (%s, %s, %s, %s)', (msg, res, user_sentiment, bot_sentiment))
    db_connection.commit()
    
    response = {
        "user_input": msg,
        "bot_response": res,
        "user_sentiment": user_sentiment,
        "bot_sentiment": bot_sentiment
    }
    
    return jsonify(response)

# Define a route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Define a route for handling user input and getting bot responses
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run()
