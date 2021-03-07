from dotenv import load_dotenv
load_dotenv()

import loggingconfig
from model import analyze_texts
from twitter_api import get_tweets_for_query, process_tweets
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
from flask import jsonify, request
import flask
import tensorflow as tf



app = flask.Flask(__name__)
CORS(app)

DEFAULT_THRESHOLD = 0.6


@app.route("/", methods=["GET"])
def home():
    return "Home"


@app.route("/searchandanalyze", methods=["GET"])
def analyze_tweets():
    try:
        query = request.args.get('q')
        if not query:
            return jsonify({
                "results": None,
                "error": True,
                "error_message": "Query not passed"
            })
        app.logger.info(f"query:{query}")
        tweets = get_tweets_for_query(query)
        if not tweets or len(tweets) == 0:
            return jsonify({
                "results": None,
                "error": True,
                "error_message": "Failed to fetch tweets"
            })

        results = analyze_texts(tweets)
        response = []

        threshold = DEFAULT_THRESHOLD
        count = 0
        for text, result in zip(tweets, results):
            isPositive = result[0] > threshold
            if isPositive:
                count+=1
            response.append(
                {"text": text,
                "result": result[0],
                "isPositive": isPositive}
            )

        return jsonify({"results": {
            "percentage": count/len(response), 
            "tweets": response
        }})
    except Exception as e:
        app.logger.error(f"Failed to analyze {e}")

@app.route("/tweets", methods=["GET"])
def tweets():
    query = request.args.get('q')
    if not query:
        raise Exception('Query is required')
        response.status_code = 400
    tweets = get_tweets_for_query(query)
    return jsonify({"tweets": tweets})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        content = request.json
        texts = content["texts"]
        results = analyze_texts(texts)
        response = []
        app.logger.info(f"texts: {texts}")
        threshold = DEFAULT_THRESHOLD
        if "threshold" in content:
            threshold = int(content["threshold"])

        for text, result in zip(texts, results):
            response.append(
                {"text": text, "result": result[0],
                    "isPositive": result[0] > threshold}
            )

        return jsonify({"results": response})
    except Exception as e:
        app.logger.error(f"Failed to analyze text {e}")


if __name__ == '__main__':
    app.run()
