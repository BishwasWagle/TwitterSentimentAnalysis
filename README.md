# Twitter Sentiment Analysis with Apache Spark and Ray

## Overview
This project analyzes Twitter data using Apache Spark and Ray to uncover insights about sentiment, trends, and user engagement patterns. Focusing on tweets during the COVID-19 period, we implemented various analytics use cases and developed a machine learning-based sentiment analysis model with hyperparameter tuning using Ray.

## Features
- **Data Storage & Retrieval**: Efficient storage and querying of Twitter JSON data using Apache Spark DataFrames
- **Sentiment Analysis**: Two implementation approaches:
  - Rule-based sentiment analysis using NLTK VADER
  - Machine learning-based sentiment analysis with TF-IDF and Random Forest, optimized using Ray Tune
- **Trend Detection**: Analysis of hashtag frequencies to identify trending topics
- **User Engagement Analysis**: Identification of influential users based on tweet counts, retweets, and favorites
- **Temporal Analysis**: Analysis of tweet patterns by hour of day
- **Language Distribution**: Analysis of language usage across the dataset
- **Complex Trend Analysis**: Cross-analysis of time and language patterns
- **Word Frequency Analysis**: Visualization of common terms using word clouds

## Repository Structure
```
TwitterSentimentAnalysis/
├── out.json # Twitter dataset (not included in repo)   
├── output/
|   └── ray_model_graphs/           # Generated Model training metrics graphs
│       ├── hyperparameter_tuning_results.png
│       ├── f1_distribution.png
│       ├── correlation_heatmap.png
│   └── visualizations/           # Generated sentiment analysis visualization
│       ├── sentiment_distribution.png
│       ├── top_hashtags.png
│       ├── tweets_by_hour.png
│       ├── language_distribution.png
│       ├── hour_language_heatmap.png
│       ├── word_cloud.png
├── models/                       # Saved ML models
├── generate_result.py            # Script to visualize ML model results
├── sentiment_analysis.py     # VADER-based sentiment analysis
├── main.py                       # Main analysis pipeline
└── sentiment_model.py    # ML-based sentiment analysis with Ray
└── README.md                     # Project documentation
```

## Technologies Used
- **Apache Spark**: For large-scale data processing and SQL queries
- **Ray**: For distributed hyperparameter tuning of machine learning models
- **NLTK**: For rule-based sentiment analysis using VADER
- **scikit-learn**: For machine learning-based sentiment analysis
- **Pandas**: For data manipulation
- **Matplotlib & Seaborn**: For data visualization
- **WordCloud**: For generating word cloud visualizations

## Key Findings
- Sentiment analysis reveals a mix of neutral, positive, and negative sentiments in COVID-19 discussions
- COVID-related hashtags dominate conversations, along with country-specific topics
- User engagement varies significantly, with some users generating exceptional retweet counts
- Tweet activity peaks during early afternoon hours (6 PM - 9 PM)
- Machine learning model achieved over 90% accuracy for sentiment classification

## Machine Learning Model
Our sentiment analysis model uses:

**TF-IDF vectorization with optimized parameters:**
- `max_features`: 3000
- `min_df`: 3
- `max_df`: 0.99

**Random Forest classifier with:**
- `n_estimators`: 100
- `max_depth`: None (unlimited)

**Performance (after tuning with Ray Tune):**
- **Accuracy**: 0.906
- **Precision**: 0.910
- **Recall**: 0.906
- **F1 Score**: 0.906

## Getting Started

### Prerequisites
- Python 3.7+
- Java 8 or 11 (for Spark)
- Apache Spark 3.4.4
- Pip packages listed in `requirements.txt`

### Installation

Clone the repository:
```bash
git clone https://github.com/BishwasWagle/TwitterSentimentAnalysis.git
cd TwitterSentimentAnalysis
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download NLTK resources:
```python
import nltk
nltk.download('vader_lexicon')
```

Place your Twitter JSON data in the parent directory.

### Running the Analysis

Run the main analysis pipeline:
```bash
python main.py
```

Train and evaluate the ML sentiment model:
```bash
python sentiment_model.py --num_samples 10 --cpus_per_trial 2
```

Visualize the ML model results:
```bash
python generate_result.py
```

## Results Visualization
Visualizations generated in `output/visualizations/`:
- Sentiment distribution
- Top hashtags
- Tweet activity by hour
- Language distribution
- Tweet distribution by hour and language (heatmap)
- Word cloud of common terms

Visualizations generated in `output/ray_model_graphs/`:
- Model performance metrics
- Hyperparameter correlation heatmap
- F1 score distribution


## Contributors
- **Bishwas Wagle**
- **Dilip Thakur**

## Acknowledgments
- University of Missouri-Columbia  
- Course: CMP_SC 8540: Principles of Big Data Management  
- Instructor: Praveen Rao

## References
- [Apache Spark documentation](https://spark.apache.org/docs/latest/)
- [Ray documentation](https://docs.ray.io/)
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
- [scikit-learn documentation](https://scikit-learn.org/)
