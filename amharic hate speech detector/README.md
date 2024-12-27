# **Amharic Hate Speech Detection API**

This project is an AI-powered web application designed to detect hate speech in Amharic text. It provides a REST API for backend integration and an intuitive frontend for user interaction. The application leverages a fine-tuned BERT model to classify Amharic text as either "Hate Speech" or "Free Speech."


## **Features**

- **AI Model Integration**: Fine-tuned BERT model for accurate hate speech classification in Amharic text.
- **REST API**: Provides endpoints to interact with the model for prediction.
- **User-Friendly Interface**: A frontend web interface for submitting text and viewing predictions.
- **Error Handling**: Ensures seamless user experience with clear error messages for invalid inputs.


## **Technologies Used**

- **Backend**: Django, Django REST Framework (DRF)
- **Frontend**: HTML, CSS, JavaScript (with Fetch API)
- **AI Model**: Hugging Face Transformers (BERT)
- **Python Environment**: Virtual Environment (`venv`)
- **Database**: SQLite (for potential user management or logs)


## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/amharic-hate-speech-detector.git
cd amharic-hate-speech-detector
```

### **2. Set Up Python Virtual Environment**
```bash
python -m venv amharic
source amharic/bin/activate    # For Linux/Mac
amharic\Scripts\activate       # For Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Prepare the AI Model**
- Place the fine-tuned BERT model in the `api/ml_model/` directory.
- Ensure the model files are structured as follows:
  ```
  api/ml_model/
  ├── tokenizer/
  │   ├── config.json
  │   ├── vocab.txt
  ├── saved_model/
      ├── model.safetensors
      ├── config.json
  ```

### **5. Run Database Migrations**
```bash
python manage.py makemigrations
python manage.py migrate
```

### **6. Start the Development Server**
```bash
python manage.py runserver
```

Access the application at [http://127.0.0.1:8000](http://127.0.0.1:8000).


## **API Endpoints**

### **Predict Text**
**Endpoint**: `/api/predict/`  
**Method**: `POST`  
**Request Body**:  
```json
{
    "text": "Your Amharic text here"
}
```
**Response**:  
```json
{
    "text": "Your Amharic text here",
    "prediction": {
        "value": 0 // or 1 (0 = Free Speech, 1 = Hate Speech)
    }
}
```


## **Frontend Usage**

1. Open the application in your browser.
2. Enter Amharic text in the input field.
3. Submit the text to view the prediction:
   - **Free Speech**: Displayed in green.
   - **Hate Speech**: Displayed in red.


## **Project Structure**

```
.
├── api/
│   ├── migrations/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── ml_model.py  # AI model integration
│   ├── serializers.py  # Input validation
│   ├── views.py  # API logic
│   ├── urls.py  # API routing
│
├── templates/
│   ├── index.html  # Frontend HTML
│
├── manage.py
├── requirements.txt
└── README.md
```


## **Sample Input and Output**

### **Input**:
```json
{
    "text": "ሰላም እንዴት ንው።"
}
```

### **Output**:
```json
{
    "text": "ሰላም እንዴት ንው።",
    "prediction": 0
}
```
**Displayed on Frontend**:  
> *The text is Free Speech.*


## **Future Improvements**

- Add user authentication for personalized services.
- Log user submissions and predictions for auditing and analytics.
- Deploy the application using cloud services like AWS or Heroku.
- Add multi-language support for more inclusivity.


## **Contributing**

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit (`git commit -m 'Add feature-name'`).
4. Push to your fork (`git push origin feature-name`).
5. Open a pull request.


## **Acknowledgments**

- Hugging Face for their robust transformer library.
- Django for providing a powerful backend framework.
- The Amharic-speaking community for providing valuable datasets and feedback.

