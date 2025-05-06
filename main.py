
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Twitter Sentiment Analysis with Apache Spark
Extended Version: Includes visualizations and word cloud generation.
"""

from __future__ import print_function
import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf, hour, to_timestamp
from pyspark.sql.types import StringType
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def create_spark_session():
    """Create and return a Spark session for the Twitter analysis"""
    spark = SparkSession.builder \
        .appName("TwitterSentimentAnalysis") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.pythonUDF.arrow.enabled", "false") \
        .getOrCreate()
    
    # Set log level to reduce console output
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    return spark

def load_twitter_data(spark, file_path):
    """Load Twitter JSON data into a Spark DataFrame"""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    try:
        # Load the JSON file
        print(file_path)
        tweets_df = spark.read.json(file_path)
        
        # Register as a temp view for SQL queries
        tweets_df.createOrReplaceTempView("tweets")
        
        print(f"Successfully loaded {tweets_df.count()} tweets")
        print(tweets_df.select("created_at").show(5, truncate=False))
        return tweets_df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def analyze_hashtags(tweets_df):
    """Extract and analyze hashtags from tweets"""
    # Explode the hashtags array to get individual hashtags
    hashtag_df = tweets_df.select(
        col("id_str"),
        explode(col("entities.hashtags.text")).alias("hashtag")
    )
    top_hashtags = hashtag_df.groupBy("hashtag").count().orderBy(col("count").desc()).limit(20)
    print("Top Hashtags:")
    top_hashtags.show(10, False)
    return hashtag_df, top_hashtags

def analyze_user_engagement(tweets_df):
    """Analyze user engagement metrics"""
    from pyspark.sql.functions import when
    user_engagement_df = tweets_df.select(
        col("user.screen_name").alias("username"),
        col("user.followers_count").alias("followers"),
        col("user.friends_count").alias("following"),
        col("retweet_count"),
        col("favorite_count")
    ).withColumn(
        "engagement_rate",
        (col("retweet_count") + col("favorite_count")) / when(col("followers") > 0, col("followers")).otherwise(1)
    )
    top_users = user_engagement_df.orderBy(col("engagement_rate").desc()).limit(10)
    print("Top Engaged Users:")
    top_users.show(10, False)
    return user_engagement_df, top_users

def temporal_analysis(tweets_df):
    """Analyze tweets by time"""
    # Convert created_at to timestamp
    tweets_with_time = tweets_df.withColumn(
        "created_timestamp",
        to_timestamp(col("created_at"), "EEE MMM dd HH:mm:ss Z yyyy")
    ).withColumn(
        "hour_of_day",
        hour(col("created_timestamp"))
    )
    hour_language_df = tweets_with_time.groupBy("hour_of_day", "lang").count()
    tweet_counts_by_hour = tweets_with_time.groupBy("hour_of_day").count().orderBy("hour_of_day")
    print("Tweet Counts by Hour:")
    tweet_counts_by_hour.show(24)
    return tweets_with_time, tweet_counts_by_hour, hour_language_df

def language_analysis(tweets_df):
    """Analyze tweet language distribution"""
    # Count tweets by language
    language_counts = tweets_df.groupBy("lang").count().orderBy(col("count").desc())
    print("Language Distribution:")
    language_counts.show(10)
    return language_counts

def word_cloud_analysis(spark):
    word_freq_df = spark.sql("""
        SELECT word, COUNT(*) AS count
        FROM (
            SELECT EXPLODE(SPLIT(LOWER(full_text), ' ')) AS word
            FROM tweets
        )
        WHERE word != '' AND LENGTH(word) > 2
        GROUP BY word
        ORDER BY count DESC
        LIMIT 100
    """)
    return word_freq_df.toPandas()


def analyze_sentiment(tweets_df):
    """Analyze sentiment of tweets"""
    from src.sentiment_analysis import get_sentiment
    
    # Register UDF for sentiment analysis
    sentiment_udf = udf(get_sentiment, StringType())
    
    # Apply sentiment analysis
    sentiment_df = tweets_df.withColumn("sentiment", sentiment_udf(col("full_text")))
    
    # Count sentiments
    sentiment_counts = sentiment_df.groupBy("sentiment").count()
    
    print("Sentiment Distribution:")
    sentiment_counts.show()
    
    return sentiment_df, sentiment_counts

def generate_visualizations(sentiment_counts, top_hashtags, tweet_counts_by_hour, language_counts, word_freq_pd, hour_language_df):
    vis_dir = "output/visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    if sentiment_counts is not None:
        sentiment_counts_pd = sentiment_counts.toPandas()
        plt.figure(figsize=(10, 6))
        sns.barplot(x="sentiment", y="count", data=sentiment_counts_pd, hue="sentiment", dodge=False, palette=["red", "gray", "green"])
        plt.title("Sentiment Distribution in Tweets")
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/sentiment_distribution.png")
        plt.close()

    top_hashtags_pd = top_hashtags.toPandas()
    plt.figure(figsize=(12, 8))
    sns.barplot(x="count", y="hashtag", data=top_hashtags_pd, hue="hashtag", dodge=False, palette="viridis")
    plt.title("Top 20 Hashtags from Tweets")
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/top_hashtags.png")
    plt.close()

    tweet_counts_by_hour_pd = tweet_counts_by_hour.toPandas()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="hour_of_day", y="count", data=tweet_counts_by_hour_pd, marker="o", linewidth=2)
    plt.title("Tweet Activity by Hour of Day")
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/tweet_activity_by_hour.png")
    plt.close()

    hour_language_pd = hour_language_df.toPandas()
    pivot_table = hour_language_pd.pivot(index="lang", columns="hour_of_day", values="count").fillna(0)    # Get top 10 languages from previously computed language_counts
    top_languages = language_counts.limit(10).toPandas()["lang"].tolist()
    pivot_table = pivot_table.loc[pivot_table.index.isin(top_languages)]
    # Create the heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_table, cmap="viridis", annot=True, fmt="g", linewidths=.5)
    plt.title("Tweet Counts by Hour of Day and Language")
    plt.xlabel("Hour of Day")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.savefig("output/visualizations/hour_language_heatmap.png")
    plt.close()

    language_counts_pd = language_counts.limit(10).toPandas()
    plt.figure(figsize=(10, 10))
    plt.pie(language_counts_pd["count"], labels=language_counts_pd["lang"], autopct="%1.1f%%",
            startangle=90, shadow=True, explode=[0.1 if i == 0 else 0 for i in range(len(language_counts_pd))])
    plt.axis("equal")
    plt.title("Language Distribution of Tweets")
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/language_distribution.png")
    plt.close()

    if word_freq_pd is not None:
        word_freq_dict = dict(zip(word_freq_pd["word"], word_freq_pd["count"]))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq_dict)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/word_cloud.png")
        plt.close()

def main():
    start_time = time.time()
    spark = create_spark_session()
    data_path = "../../out.json"  # Update this path to your data file location
    tweets_df = load_twitter_data(spark, data_path)
    hashtag_df, top_hashtags = analyze_hashtags(tweets_df)
    user_engagement_df, top_users = analyze_user_engagement(tweets_df)
    tweets_by_hour, tweet_counts_by_hour, hour_language_df = temporal_analysis(tweets_df)
    language_counts = language_analysis(tweets_df)
    word_freq_pd = word_cloud_analysis(spark)
    sentiment_df, sentiment_counts = analyze_sentiment(tweets_df)
    generate_visualizations(sentiment_counts, top_hashtags, tweet_counts_by_hour, language_counts, word_freq_pd, hour_language_df)

    spark.stop()
    print(f"Pipeline completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
