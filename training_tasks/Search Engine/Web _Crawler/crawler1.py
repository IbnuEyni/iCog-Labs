import csv
import json
import os
import random
import string
import time
import requests
from bs4 import BeautifulSoup
import queue
import re
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

nltk.download('stopwords')
nltk.download('punkt')
stopwords_list = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_name(name):
    name = re.sub(r'[^\x00-\x7F]+', ' ', name)
    name = name.lower()
    name = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return ' '.join(lemmatizer.lemmatize(word) for word in nltk.word_tokenize(name) if word not in stopwords_list)

def extract_price(price_str):
    try:
        price = re.sub(r"[^\d.]", "", price_str)
        return float(price) if price else None
    except ValueError:
        return None

def name_similarity(name1, name2):
    return SequenceMatcher(None, name1, name2).ratio()

def load_cached_products(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_cached_products(products, cache_file):
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(products, f, ensure_ascii=False, indent=4)

target_name = input("Enter the name of the target product: ")
try:
    target_price = float(input("Enter the price of the target product: "))
except ValueError:
    print("Invalid price. Please enter a valid numeric value.")
    exit()

try:
    num_similar = int(input("How many similar products do you want to retrieve? "))
except ValueError:
    print("Invalid number. Please enter a valid integer value.")
    exit()
 
cache_file = 'products_cache.json'
products = load_cached_products(cache_file)

if not products:
    urls = queue.PriorityQueue()
    urls.put((0.5, "https://www.scrapingcourse.com/ecommerce/"))
    visited_urls = []

    while not urls.empty() and len(visited_urls) < 50:
        _, current_url = urls.get()
        
        try:
            response = requests.get(current_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error encountered: {e}. Skipping this URL.")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        visited_urls.append(current_url)

        link_elements = soup.select("a[href]")
        for link_element in link_elements:
            url = link_element["href"]
            if not url.startswith("http"):
                url = requests.compat.urljoin(current_url, url)
            
            if url not in visited_urls and url not in [item[1] for item in urls.queue]:
                priority_score = 1
                if re.match(r"^https://www\.scrapingcourse\.com/ecommerce/page/\d+/?$", url):
                    priority_score = 0.5
                urls.put((priority_score, url))
        
        time.sleep(random.uniform(1, 3))

        product = defaultdict(str)
        product["url"] = current_url
        try:
            image_element = soup.find("img")
            product["image"] = image_element["src"] if image_element else "N/A"
            
            name_element = soup.select_one(".product-name")
            product["name"] = name_element.text.strip() if name_element else "N/A"
            product["name"] = preprocess_name(product["name"])
            
            price_element = soup.select_one(".price")
            product["price"] = price_element.text.strip() if price_element else "N/A"
            product["price_value"] = extract_price(product["price"])
        except Exception as e:
            print(f"Error parsing {current_url}: {e}")
            continue 
        
        products.append(product)

    save_cached_products(products, cache_file)

with open('products.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["URL", "Image", "Name", "Price"])
    for product in products:
        writer.writerow(product.values())

def find_similar_products(target_name, target_price, products, num_similar):
    processed_target_name = preprocess_name(target_name)
    similar_products = []
   
    for product in products:
        name_score = name_similarity(processed_target_name, product["name"])
        
        price_value = product.get("price_value")
        if price_value is None:
            price_score = 0  
        else:
            price_score = 1 - abs((price_value - target_price) / target_price) if target_price > 0 else 0
            
        overall_score = (name_score * 0.7) + (price_score * 0.3)
        similar_products.append((overall_score, product))

    similar_products.sort(key=lambda x: x[0], reverse=True)

    if not similar_products:
        print("\nNo similar products found.")
        return []

    print(f"\nTop {num_similar} similar products found:")
    for i, (score, product) in enumerate(similar_products[:num_similar]):
        print(f"\nRank {i + 1}:")
        print(f"Name: {product['name']}")
        print(f"Price: {product['price']}")
        print(f"URL: {product['url']}")
        print(f"Similarity Score: {score:.2f}")

    return similar_products[:num_similar]

find_similar_products(target_name, target_price, products, num_similar)
