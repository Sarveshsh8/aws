import os
import logging
import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use temporary credentials directly
assumed_role_session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN")
)

bedrock_client = assumed_role_session.client("bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id=os.getenv("EMBED_MODEL_ID"),
    client=bedrock_client
)

class TextToSQL:
    def __init__(self, csv_path: str, pdf_path: str, faiss_index_path: str):
        self.csv_path = csv_path
        self.pdf_path = pdf_path
        self.faiss_index_path = faiss_index_path
        self.embeddings = bedrock_embeddings
        self.vectorstore = None
        self.docs = []
        self.load_csv_documents()
        self.load_pdf_documents()
        self.create_vectorstore()
        self.load_vectorstore()
        self.session = boto3.Session(profile_name="test")
        self.client = self.session.client("bedrock-runtime")

    def load_csv_documents(self):
        """Loads CSV data and adds metadata."""
        if self.csv_path:
            loader = CSVLoader(file_path=self.csv_path, csv_args={"delimiter": ",", "quotechar": '"'})
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
            split_docs = text_splitter.split_documents(documents)
            
            for doc in split_docs:
                doc.metadata["source"] = "csv"
            
            self.docs.extend(split_docs)

    def load_pdf_documents(self):
        """Loads PDF data and adds metadata (NO OCR REQUIRED)."""
        if self.pdf_path:
            loader = PyMuPDFLoader(self.pdf_path)  # No OCR required
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
            split_docs = text_splitter.split_documents(documents)
            
            for doc in split_docs:
                doc.metadata["source"] = "pdf"
            
            self.docs.extend(split_docs)

    def create_vectorstore(self):
        """Creates and saves the FAISS vector store."""
        try:
            sample_text = self.docs[0].page_content
            sample_embedding = self.generate_embedding(sample_text)
            
            if sample_embedding is None or len(sample_embedding) == 0:
                logger.error("Embeddings are not generated correctly.")
                raise ValueError("Embeddings are not generated correctly.")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

        self.vectorstore = FAISS.from_documents(self.docs, self.embeddings)
        self.vectorstore.save_local(self.faiss_index_path)

    def generate_embedding(self, text):
        """Generates embeddings for the given text."""
        return self.embeddings.embed_query(text)

    def load_vectorstore(self):
        """Loads the FAISS vector store from disk."""
        self.vectorstore = FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)

    def retrieve_context(self, query: str, k: int = 10):
        """Retrieves relevant context from both CSV (for SQL) and PDF (for explanations)."""
        
        retrieved_docs = self.vectorstore.similarity_search(query, k=k)
        
        csv_content = "\n".join([doc.page_content for doc in retrieved_docs if doc.metadata.get("source") == "csv"])
        pdf_content = "\n".join([doc.page_content for doc in retrieved_docs if doc.metadata.get("source") == "pdf"])

        return pdf_content, csv_content
    
    def generate_sql_query(self, query: str):
        """Generates an SQL query based on the retrieved context and user query."""
        pdf_context, csv_context = self.retrieve_context(query)
        json_output ={
            "Answer": "To analyze the gender distribution across the data, we can generate a count of students by gender and visualize it in a bar chart.",
            "SQL Query": "SELECT gender, COUNT(*) as count FROM students GROUP BY gender",
            "SQL Query Answer": "male, 8\nfemale, 4",
            "Visualization": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "description": "Gender Distribution",
                "data": {
                "values": [
                    {"gender": "male", "count": 8},
                    {"gender": "female", "count": 4}
                ]
                },
                "mark": "bar",
                "encoding": {
                "x": {"field": "gender", "type": "nominal"},
                "y": {"field": "count", "type": "quantitative"},
                "color": {"field": "gender", "type": "nominal"}
        }
        }
        }       # Convert json_output to a JSON string
        json_output_str = json.dumps(json_output, indent=2)
        json_output_str = json_output_str.replace("{", "{{").replace("}", "}}")

        # Create the prompt for Claude 3.5 Sonnet
        prompt = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": """You are an expert AI assistant for Text-to-SQL conversion.

            **Task:**
            - Analyze the provided PDF for context and the CSV data structure to understand the schema and answer the user's question:
            1. If it requires data retrieval, generate a SQL query and summarize the answer using the PDF context.
            2. If it doesn't require SQL querying, provide only an explanation and set the SQL query to null.
            3. If the question involves time-based data or any data that would benefit from visualization (especially date ranges, timestamps, or categorical distributions), also generate a Vega-Lite specification.""",
            
            
            "messages": [
                {
                    "role": "user",
                    "content": f"""**Instructions:**
            Format your response ONLY as a valid JSON object with these fields:
            - Answer: Your explanation based on the PDF context
            - SQL Query: The SQL query to retrieve the requested data, or null if no query is needed
            - SQL Query Answer: The result of the SQL query in a simple format
            - Visualization: When appropriate, include a Vega-Lite JSON specification for visualization

            For Vega-Lite visualizations:
            - Use appropriate chart types based on the data (bar charts for categorical data, line charts for time series, etc.)
            - Keep the specification clean and minimal
            - For categorical distributions, use bar charts or pie charts as appropriate
            - IMPORTANT: Do not add any text before or after the JSON object. Your entire response should be only the JSON object itself.

            Example of exactly how your response should look:
            {json_output_str}"""
            
                }
            ],
            "temperature": 0.0
        }

        # Convert prompt to JSON string
        input_text = json.dumps(prompt)
        
        # Call Claude 3.5 Sonnet model
        response = self.client.invoke_model(
            modelId=os.getenv("MODEL_ID"), 
            body=input_text
        )

        # Parse the response
        response_body = json.loads(response['body'].read().decode("utf-8"))
        sql_query = response_body['content'][0]['text']
        
        return sql_query


if __name__ == "__main__":
    text2sql = TextToSQL(csv_path="data/data.csv", pdf_path="data/data.pdf", faiss_index_path="faiss_index")
    query = "Fetch the GPA of id number 1141"
    # query = "What is ID and Class"
    sql_query = text2sql.generate_sql_query(query)
    print(sql_query)