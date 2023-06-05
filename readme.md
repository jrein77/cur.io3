# cur.io: Text Processing and Question Answering App
This app allows you to upload multiple text documents (PDFs, DOCX files, TXT files, or CSVs), processes them into a knowledge base, and allows you to ask questions about the content of these documents. It uses OpenAI for creating embeddings and a question answering system.

# Installation
Make sure you have the required dependencies by running:
```
pip install -r requirements.txt
```

Also, this app requires an OpenAI API key. Add your OpenAI API key to a .env file in your project root like so:
```
OPENAI_API_KEY='your-key-here'
```

# Usage
You can run the Streamlit app with the command:
```
streamlit run app.py
```

# How It Works
* The text is extracted from the uploaded files. For PDFs, the PyPDF2 library is used. For DOCX files, python-docx is used. For TXT files, the built-in read() method is used. For CSV files, pandas is used to load the CSV data and extract the text.
* The extracted text is then split into smaller chunks using CharacterTextSplitter from the langchain library.
* These chunks of text are then used to build a knowledge base using FAISS from the langchain library, where each chunk of text is embedded using OpenAI's API.
* When a user asks a question, the app uses the OpenAI model and the load_qa_chain method to find the most similar chunks of text to the question. It then generates a response from the most similar chunks.
* The app then displays the response to the user's question.

Please note: This app will only work with an OpenAI API key. Make sure to set it up before running the app.
