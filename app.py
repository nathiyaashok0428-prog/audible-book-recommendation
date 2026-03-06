import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("data/clean_books.csv")

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

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(df["text_features"])

# ==============================
# COSINE SIMILARITY
# ==============================

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ==============================
# RECOMMEND FUNCTION
# ==============================

def recommend(book_title, top_n=5):

    idx = df[df["Book Name"] == book_title].index[0]

    scores = list(enumerate(cosine_sim[idx]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    scores = scores[1:top_n+1]

    book_indices = [i[0] for i in scores]

    return df.iloc[book_indices][["Book Name","Author","Rating"]]

# ==============================
# STREAMLIT UI
# ==============================

st.title("📚 Audible Intelligent Book Recommendation System")

st.write("Select a book to get similar book recommendations.")

book_list = df["Book Name"].sort_values().unique()

selected_book = st.selectbox(
    "Choose a Book",
    book_list
)

if st.button("Recommend"):

    recommendations = recommend(selected_book)

    st.subheader("📖 Recommended Books")

    st.dataframe(recommendations)

# ==============================
# TOP RATED BOOKS
# ==============================

st.subheader("⭐ Top Rated Books")

top_books = df.sort_values("Rating", ascending=False).head(10)

st.dataframe(top_books[["Book Name","Author","Rating"]])