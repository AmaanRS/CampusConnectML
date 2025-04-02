import os
import pandas as pd
import json
import joblib
from dotenv import load_dotenv
from pymongo import MongoClient
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from models.tag_recommender import TagRecommender

load_dotenv()
app = Flask(__name__)

db_url = os.getenv("MONGODB_URL")
client = MongoClient(db_url)

def cleanData():
    db = client["CampusConnectSelf"]
    collection = db["committeemodels"]

    committees = list(collection.find({}))

    df = pd.DataFrame(committees)

    df['no_of_posts'] = df['posts'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    df['followers'] = df['followers'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    df["tags"] = df["tags"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

    df = df[['committeeId', 'no_of_posts', 'followers','tags']]

    df.to_csv("2.0/dataset/formatted_committee_data.csv")

cleanData()

# Load trained model
model1 = joblib.load("2.0/dataset/models/final_model.pkl")
model2 = joblib.load("2.0/dataset/models/tag_recommender.pkl")


@app.route('/predict_rankings', methods=['GET'])
def predict_rankings():

    user_tags = request.args.get("tags", "")

    df = pd.read_csv("2.0/dataset/formatted_committee_data.csv")
    
    feature_cols = ['no_of_posts','followers']
    
    X = df[feature_cols]

    df.rename(columns={"committeeId": "committee_id"}, inplace=True)

    df["predicted_score"] = model1.predict(X)

    df.to_csv("2.0/dataset/linear_df.csv")

    recommendations_df = model2.recommend('science')[["committee_id","similarity_score"]]

    df = df.merge(recommendations_df, on="committee_id", how="left")

    df_merged = df.copy()

    scaler = MinMaxScaler()

    df_merged['predicted_score_norm'] = scaler.fit_transform(df_merged[['predicted_score']])

    df_merged['similarity_score_norm'] = scaler.fit_transform(df_merged[['similarity_score']])

    df_merged['combined_score'] = 0.4 * df_merged['predicted_score_norm'] + 0.6 * df_merged['similarity_score_norm']

    df_final = df_merged.sort_values('combined_score', ascending=False)
    
    return jsonify(df_final[["committee_id", "combined_score"]].to_dict(orient="records"))

if __name__ == '__main__':
    app.run(debug=True)