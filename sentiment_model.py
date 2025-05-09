"""Machine learning model for sentiment analysis using Ray."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import ray
from ray import tune
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentiment_analysis import get_sentiment

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

config = {
    "max_features": tune.choice([1000, 2000, 3000]),
    "min_df": tune.choice([1, 2, 3]),
    "max_df": tune.choice([0.9, 0.95, 0.99]),
    "n_estimators": tune.choice([100, 200, 300]),
    "max_depth": tune.choice([None, 10, 20, 30])
}

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

def prepare_data_for_ray(spark_df):
    """Convert Spark DataFrame to Pandas for Ray processing."""
    # Select only necessary columns to reduce memory footprint
    pandas_df = spark_df.select("full_text", "sentiment").toPandas()
    
    # Filter out rows with missing values
    pandas_df = pandas_df.dropna()
    
    return pandas_df

def train_sentiment_model(config, X_train, X_test, y_train, y_test):
    """Train a sentiment analysis model with the given configuration."""
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=config["max_features"],
        min_df=config["min_df"],
        max_df=config["max_df"]
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train a Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=42
    )
    clf.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print(accuracy)
    # Report metrics to Ray Tune
    tune.report(
        metrics={
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1
        }
    )
    
    # return clf, vectorizer

def main(args):
    """Run the sentiment analysis model training."""
    
    # Initialize Ray
    ray.init()
    
    # Create Spark session
    spark = create_spark_session()
    # Load Twitter data
    tweets_df = load_twitter_data(spark, "../../out.json")
    # Run sentiment analysis to get sentiment labels
    sentiment_udf = udf(lambda text: get_sentiment(text), StringType())
    sentiment_df = tweets_df.withColumn("sentiment", sentiment_udf(col("full_text")))
    
    # Convert to Pandas for Ray
    pandas_df = prepare_data_for_ray(sentiment_df)
    
    # Prepare the data
    X = pandas_df["full_text"]
    y = pandas_df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define hyperparameter search space
    search_space = {
        "max_features": tune.choice([1000, 2000, 3000]),
        "min_df": tune.choice([1, 2, 3]),
        "max_df": tune.choice([0.9, 0.95, 0.99]),
        "n_estimators": tune.choice([100, 200, 300]),
        "max_depth": tune.choice([None, 10, 20, 30])
    }
    
    # Define training function for Ray Tune
    def training_function(config):
        train_sentiment_model(config, X_train, X_test, y_train, y_test)
    
    # Run hyperparameter tuning
    analysis = tune.run(
        training_function,
        config=search_space,
        num_samples=args.num_samples,
        resources_per_trial={"cpu": args.cpus_per_trial},
        metric="f1",
        mode="max"
    )
    
    # Get the best configuration
    best_config = analysis.get_best_config(metric="f1", mode="max")
    print("Best hyperparameters:", best_config)
    
    # Train the final model with the best configuration
    vectorizer = TfidfVectorizer(
        max_features=best_config["max_features"],
        min_df=best_config["min_df"],
        max_df=best_config["max_df"]
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    final_model = RandomForestClassifier(
        n_estimators=best_config["n_estimators"],
        max_depth=best_config["max_depth"],
        random_state=42
    )
    final_model.fit(X_train_tfidf, y_train)
    
    # Evaluate the final model
    y_pred = final_model.predict(X_test_tfidf)
    print("Final model performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Recall:", recall_score(y_test, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
    
    # Save the model if requested
    if args.save_model:
        import joblib
        
        model_dir = os.path.join(project_root, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "sentiment_model.joblib")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
        
        joblib.dump(final_model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model using Ray Tune.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of hyperparameter samples to try")
    parser.add_argument("--cpus_per_trial", type=int, default=2, help="CPUs to allocate per trial")
    parser.add_argument("--save_model", action="store_true", help="Save the final model")
    
    args = parser.parse_args()
    main(args)