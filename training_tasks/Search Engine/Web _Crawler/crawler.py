# import requests

# # Define the URL for the ISS (International Space Station) location API
# URL = "http://api.open-notify.org/iss-now.json"

# # Send a GET request to the API and store the response
# response = requests.get(URL)

# # Print the HTTP status code of the response to check if the request was successful
# if response.status_code == 200:
#     data = response.json()

#     # Print the parsed data (ISS location details)
#     print("ISS Location Data:")
#     print(data)
# else:
#     print(
#         f"Error: Failed to retrieve data. Status code: {response.status_code}")

# image_url = "https://media.geeksforgeeks.org/wp-content/uploads/20230505175603/100-Days-of-Machine-Learning.webp"
# output_filename = "gfg_logo.png"

# response = requests.get(image_url)

# if response.status_code == 200:
#     with open(output_filename, "wb") as file:
#         file.write(response.content)
#     print(f"Image downloaded successfully as {output_filename}")
# else:
#     print("Failed to download the image.")


# Python3 program for a word frequency
#pip install requests beautifulsoup4
import requests
from bs4 import BeautifulSoup
from collections import Counter

def start(url):
    # Fetch the webpage using the URL
    source_code = requests.get(url).text
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(source_code, 'html.parser')
    # Initialize an empty list to store the words
    wordlist = []
    
    # Find all 'div' elements with the 'entry-content' class and extract text
    for each_text in soup.findAll('div'):
        content = each_text.text
        
        # Convert text to lowercase and split into words
        words = content.lower().split()
        
        # Add words to the wordlist
        for each_word in words:
            wordlist.append(each_word)
    
    # Clean the word list and create the frequency dictionary
    clean_wordlist(wordlist)

def clean_wordlist(wordlist):
    # Initialize an empty list to store clean words
    clean_list = []
    
    # Define symbols to be removed from words
    symbols = "!@#$%^&*()_-+={[}]|\\;:\"<>?/.,"
    
    # Clean the words in the wordlist
    for word in wordlist:
        for symbol in symbols:
            word = word.replace(symbol, '')
        
        # Only add non-empty words to the clean list
        if len(word) > 0:
            clean_list.append(word)
    
    # Create a dictionary and count word frequencies
    create_dictionary(clean_list)

def create_dictionary(clean_list):
    # Create a Counter object to count word frequencies
    word_count = Counter(clean_list)
    
    # Get the 10 most common words
    top = word_count.most_common(10)
    
    # Print the top 10 most common words
    print("Top 10 most frequent words:")
    print(top)
    for word, count in top:
        print(f'{word}: {count}')

if __name__ == "__main__":
    # Replace the URL with the webpage you want to scrape
    url = 'https://example.com'
    
    # Call the start function with the URL
    start(url)
