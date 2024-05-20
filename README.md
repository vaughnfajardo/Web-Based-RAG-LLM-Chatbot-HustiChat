# HustiChat

HustiChat is a conversational AI chatbot designed to facilitate easy access to Philippines Supreme Court jurisprudence. Built with Streamlit and LangChain, it integrates advanced AI models from OpenAI and Google to provide responsive and context-aware interactions based on legal documents.

## Features

- Conversational AI leveraging OpenAI and Google Generative AI models.
- Context retrieval from an online database of Philippines Supreme Court decisions.
- User-friendly web interface powered by Streamlit for seamless interactions.

## Installation

To get started with HustiChat, follow these steps:

1. Clone the repository:
```
   git clone 
   cd hustichat
   ```

2. Install the necessary dependencies:
```
pip install -r requirements.txt
```

3. Configure the required environment variables in a .env file:
```
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage
To run the application, execute the following command:
```
streamlit run your_script.py
```
Visit http://localhost:8501 in your web browser to start interacting with the chatbot.

## How to Use HustiChat
1. Input the GR number and select the month and year to initialize the chat context with relevant jurisprudence.
2. Use the chat interface to send your queries and receive contextually aware responses.
3. The application maintains a session history to enhance response accuracy and relevance.

