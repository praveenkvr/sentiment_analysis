from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import re
import tweepy

import nltk
nltk.download('stopwords')


stp_wrds = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

consumer_key = os.getenv('API_KEY')
consumer_secret = os.getenv('API_SECRET')
access_key = os.getenv('ACCESS_TOKEN')
access_secret = os.getenv('ACCESS_TOKEN_SECRET')


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


def get_tweets_for_query(query, numtweets=100):
    try:
        print(query)
        tweets = tweepy.Cursor(api.search, q=[
            f"{query} -filter:retweets"], lang="en", tweet_mode='compat').items(numtweets)

        list_tweets = [tweet.text for tweet in tweets if not tweet.retweeted and (
            'RT ' not in tweet.text)]
        return list_tweets
    except Exception as e:
        print(e)
        return None


def process_tweets(tweets):
    processed = [format_tweet(tweet) for tweet in tweets if tweet is not None]
    return processed


def clean_data(text):
    text = re.sub(r'@[^\s]+', '', text, re.UNICODE)  # remove @...
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))',
                  ' ', text, re.UNICODE)  # remove www
    # remove special characters
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = re.sub(r'[0-9]+', '', text, re.UNICODE)  # remove numbers
    return text


def process_text(text):
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stp_wrds]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = [lemmatizer.lemmatize(word, 'v') for word in text]
    text = " ".join(text)
    return text


def format_tweet(tweet):
    cleaned = clean_data(tweet)
    processed = process_text(cleaned)
    return processed
