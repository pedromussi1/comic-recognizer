<h1>Code Breakdown</h1>

<p>The purpose of this program is to identify text from an image and find the book that contains this text using the Google Books API. The program combines Optical Character Recognition (OCR) techniques with the Google Books API to achieve this functionality.</p>

<h2>Text Extraction from Image</h2>

<p>The extract_text function handles the OCR process using the Tesseract OCR engine. The process involves:</p>

<p>Reading the Image: The image is read using OpenCV.</p>

<p>Grayscale Conversion: The image is converted to grayscale to improve OCR accuracy.</p>

<p>Text Extraction: Tesseract OCR is used to extract text from the grayscale image.</p>

```py
import cv2
import pytesseract

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR using Tesseract
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray, config=custom_config)

    return text.strip()

```

<h2>Book Identification using Google Books API</h2>

<p>The get_best_match function is responsible for identifying the book that best matches the extracted text. The process involves:</p>

<p>Sending a Request to Google Books API: The extracted text is used as a query to search for books.</p>

<p>Processing the Response: The response is parsed to find the book with the highest relevance.</p>

<p>Returning the Best Match: The title and authors of the best matching book are returned.</p>

```py
import requests

def search_book(text):
    url = 'https://www.googleapis.com/books/v1/volumes'
    params = {
        'q': text,
        'maxResults': 10,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('items', [])
    else:
        print(f"Error: {response.status_code}")
        return []

def get_best_match(extracted_text):
    books = search_book(extracted_text)
    if not books:
        return "No matches found."

    # Find the best match based on the highest relevance
    best_match = max(books, key=lambda x: x.get('relevance', 0))

    volume_info = best_match.get('volumeInfo', {})
    title = volume_info.get('title', 'No title')
    authors = volume_info.get('authors', ['Unknown author'])
    return f"{title}, Authors: {', '.join(authors)}"

```

<h2>Main Function</h2>

<p>The main function integrates both components. It takes an image path as input, extracts text from the image, and finds the best matching book. The extracted text and the best match are printed to the console.</p>

```py

def main(image_path):
    # Extract text from the image
    extracted_text = extract_text(image_path)
    print(f"Extracted Text: \n\n{extracted_text}\n")

    # Get the best matching book using the Google Books API
    best_match = get_best_match(extracted_text)
    print(f"Best Match: {best_match}")

if __name__ == "__main__":
    # Example usage:
    image_path = r"D:\Computer Vision\OCR_Images\IMG.jpg"
    main(image_path)

```

<h2>Conclusion</h2>

<p>This program effectively combines OCR and web API technologies to identify the book corresponding to text extracted from an image. It leverages Tesseract OCR for text extraction and the Google Books API for book identification. This approach can be useful for various applications such as digitizing and cataloging printed materials.</p>

<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://i.imgur.com/GCZqyTU.jpeg" alt="BookPage" style="width: auto; height: 290px; margin: 20px;">
  <img src="https://i.imgur.com/8Ews4QR.png" alt="TranscribingImage" style="width: auto; height: 290px; margin: 20px;">
  <img src="https://i.imgur.com/gZxakIi.png" alt="TranslatingText" style="width: auto; height: 290px; margin: 20px;">

