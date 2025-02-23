# Text-to-SQL with AWS RAG and Chainlit

This project enables users to upload a PDF file, extract relevant information using AWS RAG and FAISS, and generate SQL queries from natural language questions using Chainlit.

## Features

* Upload a PDF file containing structured data.
* Automatically process and store document embeddings in FAISS.
* Retrieve relevant context using similarity search.
* Convert natural language queries into SQL queries.
* Interactive chatbot interface using Chainlit.

## Installation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/your-repo/aws_rag_txt2sql.git
   cd aws_rag_txt2sql
   ```
2. **Create a Virtual Environment** (Optional but recommended)
   ```sh
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   ```
3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## Environment Variables

Ensure you have the following environment variables set:

```sh
export AWS_ACCESS_KEY=your_aws_access_key
export AWS_SECRET_KEY=your_aws_secret_key
export AWS_REGION=your_aws_region
```

## Usage

### Running the App

```sh
chainlit run app.py
```

### Uploading a PDF

1. Open the app in your browser ([http://localhost:8000](http://localhost:8000/))
2. Upload a structured PDF file (e.g., student records, financial reports, etc.)
3. Once uploaded, ask a question like:
4. The system will generate and return an SQL query.
