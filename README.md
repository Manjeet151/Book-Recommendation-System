# 📚 Book Recommendation System

A web-based Book Recommendation System built using **Python, Flask, Pandas, Scikit-learn, and RapidFuzz**. It provides personalized book suggestions based on tags using **TF-IDF vectorization** and **cosine similarity**.

## 🚀 Live Demo
🔗 Deployed the complete system at: https://book-recommendation-system-4-01hq.onrender.com/)  
*(Free instance may take a few seconds to load)*

## ✨ Features

- 🔍 Search for a book by title (fuzzy matching supported)
- 📖 Get top 5 similar books based on tag similarity
- 🏆 View top 50 books by rating on homepage
- 🖼️ Includes book cover images, authors, and ratings
- ⚡ Fast recommendations using precomputed TF-IDF + cosine similarity

---

## 🛠️ Tech Stack

- **Backend**: Flask, Gunicorn
- **ML/NLP**: Scikit-learn, RapidFuzz
- **Frontend**: HTML + CSS (Jinja2 Templates)
- **Deployment**: Render (Free tier)

---

## 📁 Project Structure

book-recommendation-system/
├── static/
│ └── styles.css # (Optional) Custom styles
├── templates/
│ ├── index.html # Homepage template
│ └── recommend.html # Results page template
├── Books.csv # Book metadata with ratings and image URLs
├── Tags.csv # Tag names
├── Book_Tags.csv # Book-tag relationships
├── book_recommender.py # Flask app file
├── requirements.txt # Python dependencies
└── README.md


## 🧠 Recommendation Logic

- Merges `Books.csv`, `Tags.csv`, and `Book_Tags.csv`
- Generates a `combined_tags` field per book
- Applies **TF-IDF vectorization** on tags
- Computes **cosine similarity** matrix
- Uses **fuzzy matching** to handle partial/misspelled input titles
