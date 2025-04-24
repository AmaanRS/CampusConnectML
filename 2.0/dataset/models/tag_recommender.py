# Write code for recommending based on tags
# Current tags are sports,literature,science
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TagRecommender:
    def __init__(self, committees_df):
        """
        Initialize the recommender by fitting a CountVectorizer on the committees' tags.
        
        Parameters:
            committees_df (pd.DataFrame): DataFrame containing committee data.
                                          Must have a 'tags' column with tag strings.
        """
        # Work on a copy to avoid modifying the original DataFrame.
        self.committees_df = committees_df.copy()
        
        # Initialize and fit the CountVectorizer on the committees' tags.
        self.vectorizer = CountVectorizer()
        self.committee_vectors = self.vectorizer.fit_transform(self.committees_df['tags'])


    def recommend(self, user_tags):
        """
        Recommend committees based on provided user tags.
        
        Parameters:
            user_tags (str): A string containing user-provided tags (e.g., "science literature").
            
        Returns:
            pd.DataFrame: The committees DataFrame with an added 'similarity_score' column,
                          sorted in descending order by similarity.
        """
        # Transform the user's tags into a vector using the fitted vectorizer.
        user_vector = self.vectorizer.transform([user_tags])
        
        # Compute cosine similarity between the user vector and each committee's tag vector.
        cosine_similarities = cosine_similarity(user_vector, self.committee_vectors).flatten()
        
        # Add the similarity score to a copy of the committees DataFrame.
        recommendations = self.committees_df.copy()
        recommendations["similarity_score"] = cosine_similarities
        
        # Sort the DataFrame by similarity score in descending order.
        recommendations = recommendations.sort_values("similarity_score", ascending=False)
        return recommendations
