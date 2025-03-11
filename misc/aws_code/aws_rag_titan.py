import os
import logging
import boto3
from langchain.aws import BedrockEmbeddings, ChatBedrock
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.aws import BedrockLLM
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use temporary credentials directly
assumed_role_session = boto3.Session(
    aws_access_key_id="",
    aws_secret_access_key="",
    aws_session_token=""
)

bedrock_client = assumed_role_session.client("bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region="us-east-1",
    client=bedrock_client
)

class TextToSQL:
    def __init__(self, csv_path: str, faiss_index_path: str):
        self.csv_path = csv_path
        self.faiss_index_path = faiss_index_path
        self.embeddings = bedrock_embeddings
        self.vectorstore = None
        self.llm = BedrockLLM(
            model_id="900858-anthropic-claude-instant-v1",
            model_arn="arn:aws:bedrock:us-east-1:942268715197:application-inference-profile/4e1x5fvs8bmn",
            model_kwargs={
                "max_tokens_to_sample": 2000,
                "temperature": 0.0
            }
        )

        self.load_documents()
        self.create_vectorstore()
        self.load_vectorstore()
        self.session = boto3.Session(profile_name="test")
        self.client = self.session.client("bedrock-runtime")

    def load_documents(self):
        """Loads and splits the CSV document into chunks."""
        loader = CSVLoader(
            file_path=self.csv_path,
            csv_args={
                "delimiter": ",",
                "quotechar": '"'
            }
        )
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        self.docs = text_splitter.split_documents(documents=documents)

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
        """Retrieves the most relevant documents based on the query."""
        retrieved_output = self.vectorstore.similarity_search(query, k=k)
        context = "\n".join(doc.page_content for doc in retrieved_output)
        return context
    
    def generate_sql_query(self, query: str):
        """Generates an SQL query based on the retrieved context and user query."""
        context = self.retrieve_context(query)

        system = f"""You are an expert in Text-to-SQL conversion. Your task is to understand the given context and the question carefully. 
        {context}
    Ensure that the output strictly follows SQL syntax and aligns with the context. Do not include any explanations, clarifications, or comments.

        {query}"""

        input_text = json.dumps({"inputText": system})
        response = self.client.invoke_model(
            "arn:aws:bedrock:us-east-1:942786715197:application-inference-profile/clai5fvus8bm", 
            body=input_text
        )

        output = response['body'].read().decode("utf-8")
        response_json = json.loads(output)
        response_text = response_json["result"][0]["outputText"]
        return response_text


if __name__ == "__main__":
    text2sql = TextToSQL("archive/Students data.csv", "faiss_index_csv")
    query = "Fetch the GPA of id number 1141"
    sql_query = text2sql.generate_sql_query(query)
    print(sql_query)
