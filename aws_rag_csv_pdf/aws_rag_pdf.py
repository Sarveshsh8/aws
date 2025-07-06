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
import altair as alt
import pandas as pd
import vl_convert
from langchain_core.documents import Document
from dotenv import load_dotenv

from botocore.credentials import create_assume_role_refresher as carr
from botocore.credentials import DeferredRefreshableCredentials as DRC
from boto3.session import Session
import os



# Load environment variables
load_dotenv()


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
        # Get temporary AWS credentials
        access_key, secret_key, token = self.get_credentials()

        # Use assumed role credentials for Bedrock client
        self.assumed_role_session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=token,
        )

        self.bedrock_client = self.assumed_role_session.client("bedrock-runtime")
        self.bedrock_embeddings = BedrockEmbeddings(
            model_id=os.getenv("EMBED_MODEL_ID"),
            client=self.bedrock_client
        )

        # Initialize paths
        self.csv_path = csv_path
        self.pdf_path = pdf_path
        self.faiss_index_path = faiss_index_path
        self.embeddings = self.bedrock_embeddings
        self.vectorstore = None
        self.docs = []

        # Load documents and vector store
        self.load_csv_documents()
        self.load_pdf_documents()
        self.create_vectorstore()
        self.load_vectorstore()


    def get_credentials(self):
        """Assume AWS role and return temporary credentials"""
        session = Session(region_name="us-east-1")

        session._session._credentials = DRC(
            refresh_using=carr(
                session.client("sts"),
                {
                    "RoleArn": "arn:aws:iam::942286715197:role/app-bedrock-access-900858-us-east-1",
                    "RoleSessionName": "test"
                }
            ),
            method="sts-assume-role"
        )

        credentials = session.get_credentials().get_frozen_credentials()
        access_key = credentials.access_key
        secret_key = credentials.secret_key
        token = credentials.token

        # Set environment variables explicitly
        os.environ["AWS_ACCESS_KEY_ID"] = access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
        os.environ["AWS_SESSION_TOKEN"] = token
        os.environ["AWS_REGION"] = "us-east-1"

        # Verify they are set
        print("AWS_ACCESS_KEY_ID:", os.environ.get("AWS_ACCESS_KEY_ID"))
        print("AWS_SECRET_ACCESS_KEY:", os.environ.get("AWS_SECRET_ACCESS_KEY"))
        print("AWS_SESSION_TOKEN:", os.environ.get("AWS_SESSION_TOKEN"))
        print("AWS_REGION:", os.environ.get("AWS_REGION"))

        return access_key, secret_key, token

    def load_csv_documents(self):
        """Loads CSV data and adds metadata."""
        if self.csv_path:
            # loader = CSVLoader(file_path=self.csv_path, csv_args={"delimiter": ",", "quotechar": '"'})
            # documents = loader.load()
            df = pd.read_csv(self.csv_path, delimiter=",", quotechar='"')
    
            # Create Document objects from DataFrame rows
            documents = []
            
            # Option 1: If you want to use a specific column as content
            # Replace 'text_column' with your actual column name containing the text
            for i, row in df.iterrows():
                content = str(row['text_column'])  # Replace with your column name
                metadata = {'source': self.csv_path, 'row': i}
                documents.append(Document(page_content=content, metadata=metadata))
            
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

    def create_vectorstores(self):
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


    def create_vectorstore(self):
        """Loads an existing FAISS index if available; otherwise, creates and saves a new one."""
        try:
            # Check if FAISS index exists
            if os.path.exists(self.faiss_index_path):
                logger.info(f"FAISS index found at {self.faiss_index_path}, loading it...")
                self.vectorstore = FAISS.load_local(
                    self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True
                )
                logger.info("FAISS index loaded successfully.")
                return  # Correctly returning after loading

            if not self.docs:
                logger.error("Document list is empty. Cannot create vector store.")
                raise ValueError("Document list is empty.")

            sample_text = self.docs[0].page_content
            sample_embedding = self.generate_embedding(sample_text)

            if not sample_embedding or len(sample_embedding) == 0:
                logger.error("Embeddings are not generated correctly.")
                raise ValueError("Embeddings are not generated correctly.")

            if not hasattr(self, 'embeddings') or self.embeddings is None:
                logger.error("Embeddings function is not defined.")
                raise ValueError("Embeddings function is not defined.")

            # Create and save FAISS vector store
            logger.info("No existing FAISS index found. Creating a new one...")
            self.vectorstore = FAISS.from_documents(self.docs, self.embeddings)
            self.vectorstore.save_local(self.faiss_index_path)
            logger.info(f"FAISS vector store successfully saved at {self.faiss_index_path}")

        except Exception as e:
            logger.error(f"Error while handling FAISS index: {e}")
            raise


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
    
    

    def save_pie_chart(self,data, filename="dynamic_pie_chart.png", inner_radius=50):
        """
        Generates a pie chart and saves it as PNG.

        Parameters:
            data (list of dict): Data format [{"category": value, "count": value}]
            filename (str): Output PNG file name.
            inner_radius (int): Set to 0 for full pie, default 50 for donut chart.
        """
        df = pd.DataFrame(data["Visualization"]["data"]["values"])  # Directly use data

        chart = alt.Chart(df).mark_arc(innerRadius=inner_radius).encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color(df.columns[0] + ":N")  # Auto-detect category
        )

        png_bytes = vl_convert.vegalite_to_png(chart.to_json())
        with open(filename, "wb") as f:
            f.write(png_bytes)

        print(f"Pie chart saved as '{filename}'")

    def get_credentials(self):
        session = Session(region_name="us-east-1")


        session._session._credentials = DRC(
            refresh_using=carr(session.client("sts"),
            {
                "RoleArn": "arn:aws:iam::942286715197:role/app-bedrock-access-900858-us-east-1",
                "RoleSessionName": "test"
            }),
            method="sts-assume-role"
        )

        credentials  = session.get_credentials().get_frozen_credentials()
        access_key = credentials.access_key
        secret_key = credentials.secret_key
        token = credentials.token
        

        # Your AWS credentials
        os.environ["AWS_ACCESS_KEY_ID"] = access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
        os.environ["AWS_SESSION_TOKEN"] = token
        os.environ["AWS_REGION"] = "us-east-1"

        return access_key, secret_key, token
    
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
                    "content": f"""Explanation:\n{pdf_context}\n\nData Structure:\n{csv_context}\n\nQuestion: {query}\n\n**Instructions:**

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
        out = json.loads(sql_query)
        try:
            self.save_pie_chart(out)
        except:
            pass

       
        return sql_query
    
    def generate_sql_query_with_stream(self, query: str):
        """Generates an SQL query based on the retrieved context and user query with streaming."""
        pdf_context, csv_context = self.retrieve_context(query)
        json_output = {
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
        }
        
        # Convert json_output to a JSON string
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
                    "content": f"""Explanation:\n{pdf_context}\n\nData Structure:\n{csv_context}\n\nQuestion: {query}\n\n**Instructions:**

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
        
        # Call Claude 3.5 Sonnet model with streaming
        response_stream = self.client.invoke_model_with_response_stream(
            modelId=os.getenv("MODEL_ID"),
            body=input_text
        )
        
        # Process the streaming response
        complete_response = ""
        
        for event in response_stream["body"]:
            if "chunk" in event:
                chunk_data = json.loads(event["chunk"]["bytes"].decode("utf-8"))
                if "content" in chunk_data and len(chunk_data["content"]) > 0:
                    text_chunk = chunk_data["content"][0].get("text", "")
                    complete_response += text_chunk
                    # We yield each chunk as it arrives
                    yield text_chunk, complete_response
        
        # Try to parse the final response and save pie chart
        try:
            final_out = json.loads(complete_response)
            try:
                self.save_pie_chart(final_out)
            except:
                pass
        except json.JSONDecodeError:
            pass
    



if __name__ == "__main__":
    text2sql = TextToSQL(csv_path="data/data.csv", pdf_path="data/data.pdf", faiss_index_path="faiss_index")
    query = "Fetch the GPA of id number 1141"
    # query = "What is ID and Class"
    sql_query = text2sql.generate_sql_query(query)
    print(sql_query)