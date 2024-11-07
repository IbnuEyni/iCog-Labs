from flask import Flask, render_template, request, jsonify
import json
import os
import re
import string
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
stopwords_list = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

def preprocess_name(name):
    name = re.sub(r'[^\x00-\x7F]+', ' ', name)
    name = name.lower()
    name = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return ' '.join(lemmatizer.lemmatize(word) for word in nltk.word_tokenize(name) if word not in stopwords_list)

def load_cached_products(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def name_similarity(name1, name2):
    return SequenceMatcher(None, name1, name2).ratio()

@app.route('/', methods=['GET', 'POST'])
def home():
    products = load_cached_products('products_cache.json')
    similar_products = []

    if request.method == 'POST':
        target_name = request.form['product_name']
        try:
            target_price = float(request.form['product_price'])
        except ValueError:
            return "Invalid price. Please enter a valid numeric value."
        
        num_similar = int(request.form['num_similar'])
        processed_target_name = preprocess_name(target_name)

        for product in products:
            name_score = name_similarity(processed_target_name, product["name"])
            price_value = product.get("price_value", 0)
            if price_value is None:
                continue
            price_score = 1 - abs((price_value - target_price) / target_price) if target_price > 0 else 0.0
            overall_score = (name_score * 0.7) + (price_score * 0.3)
            similar_products.append((overall_score, product))
        
        similar_products.sort(key=lambda x: x[0], reverse=True)
        similar_products = similar_products[:num_similar]

    return render_template('index.html', products=similar_products)

if __name__ == '__main__':
    app.run(debug=True)
