import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import autogen
from autogen import AssistantAgent, UserProxyAgent
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class RAGApplication:
    """
    A Retrieval-Augmented Generation (RAG) application for querying PDF documents.
    
    This class provides a complete RAG system that can:
    - Load and process PDF documents
    - Create vector embeddings for semantic search
    - Answer questions based on retrieved document context
    - Save query history and results
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 collection_name: str = "document_collection",
                 persist_dir: str = "./chroma_db",
                 config_path: str = "OAI_CONFIG_LIST.json"):
        """
        Initialize the RAG application.
        
        Args:
            openai_api_key: OpenAI API key for embeddings and LLM
            collection_name: Name for the vector database collection
            persist_dir: Directory to store the vector database
            config_path: Path to AutoGen configuration file
        """
        self.openai_api_key = openai_api_key
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.config_path = config_path
        self.results_file = "query_results.json"
        
        # Set environment variable
        os.environ['OPENAI_API_KEY'] = self.openai_api_key
        
        # Initialize components
        self.vectordb = None
        self.assistant = None
        self.user_proxy = None
        self.embeddings = OpenAIEmbeddings()
        
        print(f"🚀 RAG Application initialized")
        print(f"📁 Collection: {self.collection_name}")
        print(f"💾 Database: {self.persist_dir}")
    
    def setup_document(self, 
                      pdf_path: str,
                      chunk_size: int = 1000,
                      chunk_overlap: int = 200) -> bool:
        """
        Load and process a PDF document for RAG.
        
        Args:
            pdf_path: Path to the PDF document
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(pdf_path):
                print(f"❌ Error: Document not found at {pdf_path}")
                return False
            
            print(f"📄 Loading document: {pdf_path}")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            print(f"✅ Loaded {len(docs)} pages from PDF")
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " "],
            )
            chunks = splitter.split_documents(docs)
            print(f"Split into {len(chunks)} chunks")
            
            # Create vector database
            print("🔍 Creating vector database...")
            self.vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name=self.collection_name
            )
            self.vectordb.persist()
            print("Vector database created and saved")
            
            return True
            
        except Exception as e:
            print(f"Error setting up document: {e}")
            return False
    
    def load_existing_database(self) -> bool:
        """
        Load an existing vector database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.persist_dir):
                print(f"No existing database found at {self.persist_dir}")
                return False
            
            print("Loading existing vector database...")
            self.vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print("Vector database loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading database: {e}")
            return False
    
    def retrieve_context(self, query: str, k: int = 5) -> Tuple[str, bool]:
        """
        Retrieve relevant context from the vector database.
        
        Args:
            query: The search query
            k: Number of chunks to retrieve
            
        Returns:
            Tuple[str, bool]: (context_text, success_flag)
        """
        try:
            if not self.vectordb:
                return "Vector database not initialized", False
            
            # Search for relevant documents
            results = self.vectordb.similarity_search(query, k=k)
            
            if not results:
                return "No relevant context found in the document", False
            
            # Combine context from all retrieved chunks
            context_parts = []
            for i, doc in enumerate(results, 1):
                context_parts.append(f"--- Context {i} ---\n{doc.page_content}\n")
            
            combined_context = "\n".join(context_parts)
            print(f"✅ Retrieved {len(results)} relevant chunks")
            
            return combined_context, True
            
        except Exception as e:
            error_msg = f"Error retrieving context: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, False
    
    def setup_agents(self, 
                    temperature: float = 0.1,
                    max_tokens: int = 1000,
                    timeout: int = 120) -> bool:
        """
        Initialize the AI agents for question answering.
        
        Args:
            temperature: LLM temperature setting
            max_tokens: Maximum tokens for responses
            timeout: Timeout for API calls
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load configuration
            config_list = autogen.config_list_from_json(self.config_path)
            llm_config = {
                "config_list": config_list,
                "timeout": timeout,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Create assistant agent
            self.assistant = AssistantAgent(
                name="Research_Assistant",
                system_message="""You are a helpful research assistant that answers questions based on provided document context.

Instructions:
1.Use Only Provided Context: Respond strictly based on the information available in the provided context. Do not rely on external knowledge or assumptions.
2.Be Thorough: Deliver a complete and well-explained answer, referencing specific details from the context.
3. Acknowledge Limitations: If the context does not contain the information needed to answer the question, clearly state: "This is not my area of expertise."
4. Quote When Appropriate: Include direct quotes from the context to support your explanation, when relevant.
5. Organize Logically: Structure your answer in a clear, logical format to enhance readability and understanding.
6. Synthesize Multiple Sources: If multiple pieces of context are provided, integrate them thoughtfully and accurately into your response.
Always provide helpful, accurate answers based on the given context.""",
                llm_config=llm_config,
            )
            
            # Create user proxy agent
            self.user_proxy = UserProxyAgent(
                name="User_Proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )
            
            print("AI agents initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error setting up agents: {e}")
            return False
    
    def ask_question(self, query: str, k: int = 5) -> Dict:
        """
        Ask a question and get an answer based on document context.
        
        Args:
            query: The question to ask
            k: Number of context chunks to retrieve
            
        Returns:
            Dict: Result containing query, answer, context, and metadata
        """
        print(f"\n🤔 Your question: {query}")
        print("🔍 Searching document...")
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": "",
            "context_preview": "",
            "success": False,
            "error": None
        }
        
        try:
            # Step 1: Retrieve context
            context, retrieval_success = self.retrieve_context(query, k=k)
            
            if not retrieval_success:
                result["answer"] = context
                result["error"] = "Context retrieval failed"
                return result
            
            # Step 2: Ensure agents are set up
            if not self.assistant or not self.user_proxy:
                if not self.setup_agents():
                    result["answer"] = "Failed to initialize AI agents"
                    result["error"] = "Agent setup failed"
                    return result
            
            # Step 3: Prepare message with context
#             message_with_context = f"""Based on the following context from the document, please answer this question: {query}
# CONTEXT:
# {context}
# QUESTION: {query}
# Please provide a comprehensive response based on the contextual information provided above.
# If the context does not supply enough detail to form a well-supported answer, simply state that ""it is outside your area of expertise"" and dont say anything else.
#  After responding, ask if there are any additional questions that need to be addressed."""

            message_with_context = f"""
You are a helpful research assistant. Please answer the following question using only the information provided in the context below.

Instructions:
1. Use only the information in the context.
2. Be thorough and cite specific details.
3. If the context doesn't contain the answer, respond with: "Hey, that’s not my area of expertise.. " and nothing else and .
4. Quote relevant parts of the context when appropriate.
5. Organize your answer clearly and logically.
6. If there are multiple pieces of context, synthesize them appropriately.

CONTEXT:
{context}

QUESTION:
{query}

Please provide a comprehensive response. If the context does not contain enough information to answer the question, say only: "This is not my area of expertise."

After answering, ask if there are any additional questions you'd like to address.
"""
            
            print("💭 Generating answer...")

    #         message_with_context = f"""Using the document context below, please respond to the following inquiry: {query}

    # DOCUMENT CONTEXT:
    # {context}

    # USER INQUIRY: {query}

    # Please deliver a thorough response drawing from the contextual information provided above. After completing your answer, ask if there are any additional questions you'd like to address."""
                
    # print("💭 Generating response...")
            
            # Step 4: Get the answer
            chat_result = self.user_proxy.initiate_chat(
                self.assistant,
                message=message_with_context,
                max_turns=1
            )
            
            # Step 5: Extract the answer
            answer = self._extract_answer(chat_result)
            
            if not answer:
                answer = "Unable to generate an answer from the retrieved context."
            
            # Update result
            result["answer"] = answer
            result["context_preview"] = context[:500] + "..." if len(context) > 500 else context
            result["success"] = True
            
            print(f"\n💡 Answer: {answer}")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            result["answer"] = error_msg
            result["error"] = str(e)
            print(f"❌ {error_msg}")
        
        # Save result
        self._save_result(result)
        return result
    
    def _extract_answer(self, chat_result) -> str:
        """Extract answer from chat result."""
        try:
            if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                for message in reversed(chat_result.chat_history):
                    if isinstance(message, dict) and message.get('name') == 'Research_Assistant':
                        return message.get('content', '').strip()
            return ""
        except Exception:
            return ""
    
    def _save_result(self, result: Dict) -> None:
        """Save query result to JSON file."""
        try:
            # Load existing results
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_results = []
            
            # Add new result
            all_results.append(result)
            
            # Save back to file
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Result saved to {self.results_file}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save result: {e}")
    
    def get_query_history(self, limit: int = 10) -> List[Dict]:
        """
        Get query history.
        
        Args:
            limit: Maximum number of recent queries to return
            
        Returns:
            List[Dict]: List of recent query results
        """
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            return results[-limit:] if limit else results
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def print_history(self, limit: int = 5) -> None:
        """Print recent query history."""
        results = self.get_query_history(limit)
        
        if not results:
            print("📋 No query history found.")
            return
        
        print(f"\n📋 Recent queries (last {len(results)}):")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Q: {result['query']}")
            print(f"   A: {result['answer'][:150]}...")
            if result.get('context_preview'):
                print(f"   Context: {result['context_preview'][:100]}...")
    
    def test_system(self) -> bool:
        """
        Test all components of the RAG system.
        
        Returns:
            bool: True if all tests pass, False otherwise
        """
        print("🧪 Testing RAG system components...")
        
        # Test 1: Vector database
        if not self.vectordb:
            print("❌ Vector database not loaded")
            return False
        print("✅ Vector database loaded")
        
        # Test 2: Context retrieval
        try:
            context, success = self.retrieve_context("test query", k=2)
            if not success:
                print(f"❌ Context retrieval failed: {context}")
                return False
            print("✅ Context retrieval working")
        except Exception as e:
            print(f"❌ Context retrieval error: {e}")
            return False
        
        # Test 3: Agent setup
        if not self.setup_agents():
            print("❌ Agent setup failed")
            return False
        print("✅ AI agents working")
        
        print("✅ All system tests passed!")
        return True
    
    def run_interactive(self) -> None:
        """Run the interactive question-answering session."""
        print("=== Interactive RAG Session ===")
        print("Commands: 'quit' to exit, 'history' for recent queries, 'test' to test system")
        
        while True:
            try:
                user_input = input("\n❓ Your question: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'test':
                    self.test_system()
                    continue
                
                elif user_input.lower() == 'history':
                    self.print_history()
                    continue
                
                elif user_input == '':
                    print("Please enter a question.")
                    continue
                
                # Ask the question
                self.ask_question(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


# Example usage and convenience functions
def create_rag_app(openai_api_key: str, 
                   collection_name: str = "my_documents") -> RAGApplication:
    """
    Create a new RAG application instance.
    
    Args:
        openai_api_key: Your OpenAI API key
        collection_name: Name for the document collection
        
    Returns:
        RAGApplication: Configured RAG application instance
    """
    return RAGApplication(
        openai_api_key=openai_api_key,
        collection_name=collection_name
    )


def main():
    """Example usage of the RAG application."""
    # Configuration
    OPENAI_API_KEY = "your-openai-api-key-here"
    PDF_PATH = "documents/your-document.pdf"
    
    # Create RAG application
    rag = RAGApplication(OPENAI_API_KEY)
    
    # Setup document (first time only)
    if not os.path.exists(rag.persist_dir):
        print("Setting up document for the first time...")
        if not rag.setup_document(PDF_PATH):
            print("Failed to setup document. Exiting.")
            return
    else:
        # Load existing database
        if not rag.load_existing_database():
            print("Failed to load existing database. Exiting.")
            return
    
    # Test system
    if not rag.test_system():
        print("System test failed. Please check configuration.")
        return
    
    # Run interactive session
    rag.run_interactive()


if __name__ == "__main__":
    # Initialize
    key = ""
    rag = RAGApplication(key)

    # Setup document (first time)
    rag.setup_document("documents/diabetes.pdf")

    # Ask questions
    result = rag.ask_question("What is a type1 diabetes?")
    print(result['answer'])