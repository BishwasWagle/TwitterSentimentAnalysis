"""Functions for sentiment analysis of tweets."""

from pyspark.sql.functions import *
from pyspark.sql.types import *
import nltk

# Download NLTK resources if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


def get_sentiment(text):
    """
    Calculate sentiment of text using VADER.
    Returns 'positive', 'negative', or 'neutral'.
    """
    if not text:
        return "neutral"
    
    # Instantiate VADER inside the function so each worker node can initialize it properly
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    if sentiment['compound'] > 0.05:
        return "positive"
    elif sentiment['compound'] < -0.05:
        return "negative"
    else:
        return "neutral"

def analyze_sentiment(tweets_df):
    """
    Perform sentiment analysis on the tweet text.
    Returns a new DataFrame with sentiment column.
    """
    # Register the sentiment UDF
    sentiment_udf = udf(get_sentiment, StringType())
    
    # Apply the UDF to get sentiment
    sentiment_df = tweets_df.withColumn(
        "sentiment", 
        sentiment_udf(col("full_text"))
    )
    
    return sentiment_df

def get_sentiment_distribution(sentiment_df):
    """
    Get the distribution of sentiments.
    Returns a DataFrame with sentiment counts.
    """
    return sentiment_df.groupBy("sentiment").count().orderBy("sentiment")