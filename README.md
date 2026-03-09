# 📚 Audible Book Recommendation System

## Project Overview

This project builds an **intelligent book recommendation system** using machine learning techniques.
The system recommends similar Audible books based on book features such as **title, author, description, and genre**.

The project demonstrates multiple recommendation strategies and compares their performance.

---

# Project Features

* Content-Based Recommendation
* Clustering-Based Recommendation
* Hybrid Recommendation System
* Natural Language Processing (TF-IDF)
* Cosine Similarity
* KMeans Clustering
* Streamlit Web Application

---

# Dataset

Two Audible datasets were used:

1. **Audible Catalog Dataset**
2. **Audible Catalog Advanced Features**

### Key Columns

* Book Name
* Author
* Rating
* Number of Reviews
* Price
* Description
* Listening Time
* Genre

---

# Project Pipeline

Raw Dataset
↓
Data Merge
↓
Data Cleaning
↓
Exploratory Data Analysis (EDA)
↓
NLP Feature Engineering
↓
Recommendation Models
↓
Model Evaluation
↓
Streamlit Web Application

---

# Recommendation Models

## 1. Content-Based Recommendation

Recommends books similar to the selected book using:

* TF-IDF Vectorization
* Cosine Similarity

Example:

User selects:

`Atomic Habits`

System recommends:

* Deep Work
* The Power of Habit
* Mindset
* The 7 Habits of Highly Effective People

---

## 2. Clustering-Based Recommendation

Algorithm used:

`KMeans Clustering`

Purpose:
Group similar books into clusters.

Example clusters:

Cluster 0 → Business Books
Cluster 1 → Self Help
Cluster 2 → Mystery
Cluster 3 → Technology

When a user selects a book, the system recommends books from the **same cluster**.

---

## 3. Hybrid Recommendation System

Hybrid approach combines:

Content Similarity + Popularity Score

Popularity score formula:

```
Popularity Score = Rating × log(Number of Reviews)
```

This ensures recommended books are:

* Similar
* Popular

---

# Model Evaluation

Clustering models are evaluated using:

### Silhouette Score

Measures cluster quality.

Range:
-1 to 1

Higher value = better clustering.

### Davies–Bouldin Score

Measures cluster separation.

Lower value = better clustering.

---

# Web Application

The system is deployed using **Streamlit**.

Features:

* Select recommendation model
* Choose a book
* Get top recommended books
* Interactive UI

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLP (TF-IDF)
* KMeans Clustering
* Streamlit

---

# Project Structure

```
audible_book_recommendation
│
├── data
│   ├── clean_books.csv
│   └── books_with_clusters.csv
│
├── scripts
│   ├── data_merge.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── nlp_features.py
│   ├── recommendation_model.py
│   ├── clustering_model.py
│   └── model_evaluation.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# Future Improvements

* Add collaborative filtering
* Add user behavior tracking
* Improve UI with book thumbnails
* Deploy scalable recommendation service
