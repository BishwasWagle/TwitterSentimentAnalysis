#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Twitter Sentiment Analysis with Apache Spark
Extended Version: Includes visualizations and word cloud generation.
Also includes user engagement analysis.
"""

from __future__ import print_function
import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf, hour, to_timestamp, sum as spark_sum
from pyspark.sql.types import StringType
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np

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
        print(tweets_df.select('created_at','lang','retweet_count','favorite_count').show(20, truncate=False))
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

def analyze_user_engagement_basic(tweets_df):
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
    print("Top Engaged Users (by engagement rate):")
    top_users.show(10, False)
    return user_engagement_df, top_users

def analyze_user_engagement_detailed(tweets_df):
    """
    Detailed user engagement analysis as described in the document.
    Analyzes tweet counts, total retweets, and total favorites per user.
    """
    # Use SQL for this query as it's cleaner and matches the document example
    user_engagement_sql = """
    SELECT
        user.screen_name AS username,
        COUNT(*) AS tweet_count,
        SUM(retweet_count) AS total_retweets,
        SUM(favorite_count) AS total_favorites
    FROM tweets
    GROUP BY user.screen_name
    ORDER BY tweet_count DESC
    LIMIT 20
    """
    
    user_engagement_detailed = tweets_df.sparkSession.sql(user_engagement_sql)
    print("Top Active Users (by tweet count):")
    user_engagement_detailed.show(10, False)
    
    # Also get top users by retweets
    top_by_retweets = tweets_df.groupBy("user.screen_name").agg(
        spark_sum("retweet_count").alias("total_retweets")
    ).orderBy(col("total_retweets").desc()).limit(10)
    
    print("Top Influential Users (by retweets):")
    top_by_retweets.show(10, False)
    
    return user_engagement_detailed, top_by_retweets

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
    from sentiment_analysis import get_sentiment
    
    # Register UDF for sentiment analysis
    sentiment_udf = udf(get_sentiment, StringType())
    
    # Apply sentiment analysis
    sentiment_df = tweets_df.withColumn("sentiment", sentiment_udf(col("full_text")))
    
    # Count sentiments
    sentiment_counts = sentiment_df.groupBy("sentiment").count()
    
    print("Sentiment Distribution:")
    sentiment_counts.show()
    
    return sentiment_df, sentiment_counts

def generate_visualizations(sentiment_counts, top_hashtags, tweet_counts_by_hour, language_counts, 
                           word_freq_pd, hour_language_df, user_engagement_detailed, top_by_retweets):
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
    # Filter pivot table to include only top languages
    if set(top_languages).issubset(set(pivot_table.index)):
        pivot_table = pivot_table.loc[pivot_table.index.isin(top_languages)]
        print(pivot_table)
        # Create the heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(pivot_table, cmap="viridis", annot=True, fmt="g", linewidths=.5)
        plt.title("Tweet Counts by Hour of Day and Language")
        plt.xlabel("Hour of Day")
        plt.ylabel("Language")
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/hour_language_heatmap.png")
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
    
    # Create user engagement visualizations
    if user_engagement_detailed is not None:
        user_engagement_pd = user_engagement_detailed.toPandas()
        
        # Take top 10 users for visualization
        top_users_pd = user_engagement_pd.head(10)
        
        # Create a figure with 3 subplots as described in the document
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        
        # Plot 1: Tweet Count
        axes[0].bar(top_users_pd['username'], top_users_pd['tweet_count'], color='skyblue')
        axes[0].set_title('Tweet Count')
        axes[0].set_xticklabels(top_users_pd['username'], rotation=90)
        axes[0].set_ylabel('Counts')
        
        # Plot 2: Total Retweets
        axes[1].bar(top_users_pd['username'], top_users_pd['total_retweets'], color='orange')
        axes[1].set_title('Total Retweets')
        axes[1].set_xticklabels(top_users_pd['username'], rotation=90)
        
        # Plot 3: Total Favorites
        axes[2].bar(top_users_pd['username'], top_users_pd['total_favorites'], color='green')
        axes[2].set_title('Total Favorites')
        axes[2].set_xticklabels(top_users_pd['username'], rotation=90)
        
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/user_engagement_metrics.png")
        plt.close()
        
        # Create a highlight visualization for users with exceptional engagement
        # Focus on retweets since they show the most dramatic differences
        top_retweets_pd = top_by_retweets.toPandas()
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(top_retweets_pd['screen_name'], top_retweets_pd['total_retweets'], color='coral')
        
        # Find the highest value to highlight it
        max_idx = top_retweets_pd['total_retweets'].idxmax()
        bars[max_idx].set_color('red')
        
        plt.title('Top Users by Total Retweets')
        plt.xlabel('Username')
        plt.ylabel('Total Retweets')
        plt.xticks(rotation=45, ha='right')
        plt.ticklabel_format(style='plain', axis='y')  # Disable scientific notation
        
        # Add value labels to bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/top_users_by_retweets.png")
        plt.close()

        # Create a scatter plot showing the relationship between tweet count and engagement
        plt.figure(figsize=(12, 8))
        
        # Calculate a normalized engagement score
        user_engagement_pd['engagement_score'] = (user_engagement_pd['total_retweets'] + 
                                                user_engagement_pd['total_favorites']) / user_engagement_pd['tweet_count']
        
        # Use log scale for better visualization if there are extreme values
        plt.scatter(user_engagement_pd['tweet_count'], 
                   np.log1p(user_engagement_pd['engagement_score']),
                   alpha=0.7, s=100, c=np.log1p(user_engagement_pd['total_retweets']), cmap='viridis')
        
        plt.title('User Activity vs. Engagement')
        plt.xlabel('Number of Tweets (Activity)')
        plt.ylabel('Log Engagement Score (Retweets + Favorites per Tweet)')
        plt.colorbar(label='Log Total Retweets')
        
        # Annotate interesting points
        for i, row in user_engagement_pd.nlargest(5, 'engagement_score').iterrows():
            plt.annotate(row['username'], 
                        (row['tweet_count'], np.log1p(row['engagement_score'])),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/user_activity_vs_engagement.png")
        plt.close()

def main():
    start_time = time.time()
    spark = create_spark_session()
    data_path = "../../out.json"  # Update this path to your data file location
    tweets_df = load_twitter_data(spark, data_path)
    
    # Perform analyses
    hashtag_df, top_hashtags = analyze_hashtags(tweets_df)
    user_engagement_df, top_users = analyze_user_engagement_basic(tweets_df)
    user_engagement_detailed, top_by_retweets = analyze_user_engagement_detailed(tweets_df)
    tweets_by_hour, tweet_counts_by_hour, hour_language_df = temporal_analysis(tweets_df)
    language_counts = language_analysis(tweets_df)
    word_freq_pd = word_cloud_analysis(spark)
    sentiment_df, sentiment_counts = analyze_sentiment(tweets_df)
    
    # Generate all visualizations
    generate_visualizations(
        sentiment_counts, 
        top_hashtags, 
        tweet_counts_by_hour, 
        language_counts, 
        word_freq_pd, 
        hour_language_df,
        user_engagement_detailed,
        top_by_retweets
    )

    spark.stop()
    print(f"Pipeline completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()