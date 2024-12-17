import os.path
import pickle
import datetime
from scipy.sparse import hstack, csr_matrix
class Sentimental:
    def __init__(self):
    # Load model and vectorizers
        pwd = os.path.dirname(__file__)
        with open(f'{pwd}\\naive_bayes_model.pkl', 'rb') as model_file:
            self.loaded_model = pickle.load(model_file)

        with open(f'{pwd}\\vectorizer_cleaned.pkl', 'rb') as vec_file_cleaned:
            self.loaded_vectorizer_cleaned = pickle.load(vec_file_cleaned)

        with open(f'{pwd}\\vectorizer_selected.pkl', 'rb') as vec_file_selected:
            self.loaded_vectorizer_selected = pickle.load(vec_file_selected)


    def find_time(self):
        text_time = [ "morning", "noon", "night" ]
        current_time = datetime.datetime.now()
        if current_time.hour < 12:
            return text_time[ 0 ]
        elif current_time.hour < 18:
            return text_time[ 1 ]
        else:
            return text_time[ 2 ]


    def predict_sentiment(self,text):
        # Transform features
        cleaned_text_vector = self.loaded_vectorizer_cleaned.transform([ text ])
        time_vector = self.loaded_vectorizer_selected.transform([ self.find_time() ])
        # Combine features
        combined_features = hstack([ cleaned_text_vector, time_vector ])
        print(f"Combined features shape: {combined_features.shape}")

        # Adjust features to match expected dimensions
        expected_features = 10001
        current_features = combined_features.shape[ 1 ]
        if current_features < expected_features:
            padding = csr_matrix((combined_features.shape[ 0 ], expected_features - current_features))
            combined_features = hstack([ combined_features, padding ])

        # Predict sentiment
        sentiment = self.loaded_model.predict(combined_features)[ 0 ]
        if sentiment == 0:
            sentiment = "negative"
        elif sentiment == 1:
            sentiment = "neutral"
        elif sentiment == 2:
            sentiment = "positive"
        return sentiment


# Test
if __name__ == "__main__":
    sentimental = Sentimental()
    print(sentimental.predict_sentiment("I am happy"))

