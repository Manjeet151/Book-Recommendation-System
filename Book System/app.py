from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

app = Flask(__name__)

# Load datasets
books = pd.read_csv("Books.csv", encoding="latin1")
tags = pd.read_csv("Tags.csv", encoding="latin1")
book_tags = pd.read_csv("Book_Tags.csv", encoding="latin1")

# Merge datasets
book_tags_merged = pd.merge(book_tags, tags, on="tag_id", how="left")
book_data = pd.merge(books, book_tags_merged, on="book_id", how="left")

# Combine tags for each book
book_data['combined_tags'] = book_data.groupby('book_id')['tag_name'].transform(lambda x: ' '.join(x))
book_data = book_data.drop_duplicates(subset=['book_id'])

# Create lowercase version of titles for fuzzy matching
book_data['lower_title'] = book_data['title'].str.lower()

# Create TF-IDF matrix from combined tags
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(book_data['combined_tags'].fillna(""))

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Function to get recommendations using fuzzy matching and similarity
def get_recommendations_by_title(title, cosine_sim=cosine_sim, top_n=5):
    lower_title = title.lower()
    all_titles = book_data['lower_title'].tolist()

    closest_title, score, idx = process.extractOne(lower_title, all_titles, score_cutoff=60)

    if closest_title is None:
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]

    return book_data.iloc[book_indices][['title', 'author', 'image-url', 'rating']]


# Route: Homepage
@app.route('/')
def home():
    try:
        top_books = books.nlargest(50, "rating")[['title', 'author', 'image-url', 'download-url', 'rating']]
    except KeyError:
        top_books = pd.DataFrame(columns=['title', 'author', 'image-url', 'rating', 'download-url'])

    return render_template('index.html', books=top_books.values)


# Route: Recommendation
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        user_input = request.form['user_input']
        recommendations = get_recommendations_by_title(user_input)

        if not recommendations.empty:
            data = recommendations.values
            message = None
        else:
            data = []
            message = "No recommendations found. Please try another title."

        return render_template('recommend.html', data=data, message=message)

    return render_template('recommend.html')

