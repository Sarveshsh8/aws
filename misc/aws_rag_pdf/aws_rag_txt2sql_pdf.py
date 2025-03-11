import os
from langchain_aws import ChatBedrock
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import os
import boto3
from langchain.embeddings import BedrockEmbeddings

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS")
AWS_REGION = os.environ.get("AWS_REGION_ID")
AWS_SECRET = os.environ.get("AWS_SECRET")


class TextToSQL:
    def __init__(self, pdf_path: str, faiss_index_path: str):
        self.llm = ChatBedrock(
            model_id="anthropic.claude-instant-v1",
            model_kwargs=dict(temperature=0),
        )
        self.pdf_path = pdf_path
        self.faiss_index_path = faiss_index_path
        self.bedrock_client = boto3.client(service_name='bedrock-runtime', 
                              region_name='us-east-1')
        self.embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=self.bedrock_client)
        self.vectorstore = None
        self.load_documents()
        self.create_vectorstore()
        self.load_vectorstore()

    def load_documents(self):
        """Loads and splits the document into chunks."""
        loader = PyPDFLoader(self.pdf_path)
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
        system = f"""You are an expert in Text-to-SQL conversion. Your task is to understand the given context and the question carefully, then generate an accurate SQL query based on the provided information.
        {context}
        Ensure that the output strictly follows SQL syntax and aligns with the context. Do not include any explanations, clarifications, or additional details—only provide the SQL query as the final output.
        """
        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        chain = prompt | self.llm
        return chain.invoke({"text": query})


# Example Usage
# if __name__ == "__main__":
#     text2sql = TextToSQL("archive/Students data.pdf", "faiss_index")
#     query = "Fetch the GPA of id number 1141"
#     sql_query = text2sql.generate_sql_query(query)
#     print(sql_query.content)
