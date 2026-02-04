import pandas as pd
import re
import joblib
#for saving the model
from sklearn.model_selection import train_test_split#for training and testing the data
from sklearn.feature_extraction.text import TfidfVectorizer#converts words to numbers for the model to understand
from sklearn.naive_bayes import MultinomialNB#What it does:Looks at word patterns,Learns which words appear with which moods,Makes predictions based on probability
from sklearn.pipeline import Pipeline#Does automatically changing words to numbers
data = {
    "text": [
        "I feel very happy today",
        "I am excited about life",
        "I feel calm and peaceful",
        "I am relaxed and content",

        "I feel sad and lonely",
        "I am tired and exhausted",
        "I feel overwhelmed and stressed",
        "Life feels heavy right now",

        "I am angry and frustrated",
        "This makes me very mad",
        "I feel irritated and annoyed",

        "I am anxious and worried",
        "I feel nervous and uneasy"
    ],
    "mood": [
        "happy", "happy", "calm", "calm",
        "sad", "sad", "stressed", "sad",
        "angry", "angry", "angry",
        "stressed", "stressed"
    ]
}

df = pd.DataFrame(data)
#cleaning the text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)






X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["mood"], test_size=0.2, random_state=42
)
#X_train → text for learning,y_train → correct moods,X_test → text for testing,y_test → answers to compare with

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])#converts word to numbers and then uses MuliinonlalNB to learn

model.fit(X_train, y_train)
#learns from the text and mood
joblib.dump(model, "mood_model.pkl")
print("Model saved successfully!")
#for saving the model