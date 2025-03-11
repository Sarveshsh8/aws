import os
from langchain_aws import ChatBedrock
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain_aws import BedrockLLM
import json

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS")
AWS_REGION = os.environ.get("AWS_REGION_ID")
AWS_SECRET = os.environ.get("AWS_SECRET")

class TextToSQL:
    def __init__(self, csv_path: str, faiss_index_path: str):
        self.llm = BedrockLLM(
            model_id="anthropic.claude-instant-v1",
            model_kwargs=dict(temperature=0),
        )
        self.bedrock_client = boto3.client(service_name='bedrock-runtime', 
                              region_name='us-east-1')
        self.csv_path = csv_path
        self.faiss_index_path = faiss_index_path
        # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=self.bedrock_client)
        self.vectorstore = None
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
                "quotechar": '"',
            },
        )
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        self.docs = text_splitter.split_documents(documents=documents)

    def create_vectorstore(self):
        """Creates and saves the FAISS vector store."""
        self.vectorstore = FAISS.from_documents(self.docs, self.embeddings)
        self.vectorstore.save_local(self.faiss_index_path)

    def load_vectorstore(self):
        """Loads the FAISS vector store from disk."""
        self.vectorstore = FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)

    def retrieve_context(self, query: str, k: int = 10):
        """Retrieves the most relevant documents based on the query."""
        retrived_output = self.vectorstore.similarity_search(query, k=k)
        content = [doc.page_content for doc in retrived_output]
        return "\n".join(content)

    # def generate_sql_query(self, query: str):
    #     """Generates an SQL query based on the retrieved context and user query."""
    #     context = self.retrieve_context(query)
        
    #     # human = "{text}"
    #     messages = [
    #         ("human","{question}"),
    #     ]
    #     prompt = ChatPromptTemplate.from_messages(messages)
    #     chain = prompt | self.llm
    #     system = f"""You are an expert in Text-to-SQL conversion. Your task is to understand the given context and the question carefully, then generate an accurate SQL query based on the provided information.
    #     {context}
    #     Ensure that the output strictly follows SQL syntax and aligns with the context. Do not include any explanations, clarifications, or additional details—only provide the SQL query as the final output.
    #     {query}
    #     """
    #     # return chain.invoke()
    #     return chain.invoke({"text": system})


    def generate_sql_query(self, query: str):
        """Generates an SQL query based on the retrieved context and user query."""
        context = self.retrieve_context(query)
        system = f"""You are an expert in Text-to-SQL conversion. Your task is to understand the given context and the question carefully, then generate an accurate SQL query based on the provided information.
        {context}
        Ensure that the output strictly follows SQL syntax and aligns with the context. Do not include any explanations, clarifications, or additional details—only provide the SQL query as the final output.
        {query}
        """
        input_text = json.dumps({"inputText":system})
        response = self.client.invoke_model(model_id = model_id,
                                            body=input_text)

        output = response['body'].read().decode('utf-8')
        model_response = json.loads(output)
        response_text = model_response["results"][0]["outputText"]
        return response_text
    
