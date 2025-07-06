import os
from langchain_aws import ChatBedrock
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain_aws import BedrockLLM

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
        self.session

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

    def generate_sql_query(self, query: str):
        """Generates an SQL query based on the retrieved context and user query."""
        context = self.retrieve_context(query)
        
        # human = "{text}"
        messages = [
            ("human","{question}"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm
        system = f"""You are an expert in Text-to-SQL conversion. Your task is to understand the given context and the question carefully, then generate an accurate SQL query based on the provided information.
        {context}
        Ensure that the output strictly follows SQL syntax and aligns with the context. Do not include any explanations, clarifications, or additional details—only provide the SQL query as the final output.
        {query}
        """
        # return chain.invoke()
        return chain.invoke({"text": system})
    
#     def generate_sql_query(self, query: str):
#         """Generates an SQL query based on the retrieved context and user query."""
#         context = self.retrieve_context(query)
#         # Create a single prompt string instead of using ChatPromptTemplate
#         # prompt_text = f"""You are an expert in Text-to-SQL conversion. Your task is to understand the given context and the question carefully, then generate an accurate SQL query based on the provided information.

#         # {context}

#         # Ensure that the output strictly follows SQL syntax and aligns with the context. Do not include any explanations, clarifications, or additional details—only provide the SQL query as the final output.

#         # Question: {query}
#         # SQL Query:"""
#         from langchain.prompts import PromptTemplate
#         prompt_text = PromptTemplate.from_template(
#         """Human: You are an expert in Text-to-SQL conversion. Your task is to understand the given context and the question carefully, then generate an accurate SQL query based on the provided information.

# {context}

# Question: {question}

# Ensure that the output strictly follows SQL syntax and aligns with the context. Do not include any explanations, clarifications, or additional details—only provide the SQL query as the final output.""")

#         # Use the BedrockLLM directly with the prompt string
#         response = self.llm.invoke(prompt_text)
#         return response


# # Example Usage
# if __name__ == "__main__":
#     text2sql = TextToSQL("archive/Students data.csv", "faiss_index_csv")
#     query = "Fetch the GPA of id number 1141"
#     sql_query = text2sql.generate_sql_query(query)
#     print(sql_query)




# model_kwargs={
#         "max_tokens_to_sample": 2000,  # This is the required parameter that's missing
#         "temperature": 0.0,
#         # other parameters as needed
#     }
