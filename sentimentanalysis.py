import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'


text_samples = [
    "I love this product, it's amazing!",
    "This movie was terrible, I hated it.",
    "The weather today is okay, not too bad.",
    "I'm feeling neutral about this article."
]

for text in text_samples:
    sentiment = analyze_sentiment(text)
    print(f"Text: '{text}' --> Sentiment: {sentiment}")
