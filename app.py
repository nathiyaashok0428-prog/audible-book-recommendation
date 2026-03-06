import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Audible Recommendation System", layout="wide")

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    df = pd.read_csv("data/books_with_clusters.csv")
    return df

df = load_data()

# ==============================
# CREATE TEXT FEATURES
# ==============================

df["text_features"] = (
    df["Book Name"].fillna("") + " " +
    df["Author"].fillna("") + " " +
    df["Description"].fillna("") + " " +
    df["Ranks and Genre"].fillna("")
)

# ==============================
# TF-IDF
# ==============================

@st.cache_data
def create_tfidf(data):

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

    matrix = tfidf.fit_transform(data)

    return matrix

tfidf_matrix = create_tfidf(df["text_features"])

# ==============================
# COSINE SIMILARITY
# ==============================

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ==============================
# CONTENT BASED RECOMMENDATION
# ==============================

def content_recommend(book_title, n=5):

    idx = df[df["Book Name"] == book_title].index[0]

    scores = list(enumerate(cosine_sim[idx]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    scores = scores[1:n+1]

    indices = [i[0] for i in scores]

    return df.iloc[indices][["Book Name","Author","Rating"]]

# ==============================
# CLUSTER RECOMMENDATION
# ==============================

def cluster_recommend(book_title, n=5):

    cluster = df[df["Book Name"] == book_title]["cluster"].values[0]

    recs = df[df["cluster"] == cluster]

    recs = recs[recs["Book Name"] != book_title]

    return recs[["Book Name","Author","Rating"]].head(n)

# ==============================
# HYBRID RECOMMENDATION
# ==============================

def hybrid_recommend(book_title, n=5):

    idx = df[df["Book Name"] == book_title].index[0]

    scores = list(enumerate(cosine_sim[idx]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    scores = scores[1:20]

    indices = [i[0] for i in scores]

    temp = df.iloc[indices].copy()

    temp["popularity"] = temp["Rating"] * np.log1p(temp["Number of Reviews"])

    temp = temp.sort_values("popularity", ascending=False)

    return temp[["Book Name","Author","Rating"]].head(n)

# ==============================
# UI
# ==============================

st.title("📚 Audible Intelligent Book Recommendation System")

st.write("Choose a recommendation model and book.")

# Model selector

model_type = st.radio(
    "Recommendation Model",
    ["Content Based","Clustering Based","Hybrid"]
)

# Book selector

book_list = df["Book Name"].sort_values().unique()

selected_book = st.selectbox("Select Book", book_list)

# Recommendation

if st.button("Recommend"):

    if model_type == "Content Based":

        recs = content_recommend(selected_book)

    elif model_type == "Clustering Based":

        recs = cluster_recommend(selected_book)

    else:

        recs = hybrid_recommend(selected_book)

    st.subheader("Recommended Books")

    st.dataframe(recs)

# Top rated books

st.subheader("⭐ Top Rated Books")

top_books = df.sort_values("Rating", ascending=False).head(10)

st.dataframe(top_books[["Book Name","Author","Rating"]])