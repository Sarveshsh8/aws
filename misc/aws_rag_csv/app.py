
import os
import chainlit as cl
from aws_rag_txt2sql_csv import TextToSQL  

# Initialize without a PDF file (we'll set it after upload)
text2sql = None  

@cl.on_message
async def handle_message(message: cl.Message):
    global text2sql

    # Check if a file is uploaded
    if message.elements:
        uploaded_file = message.elements[0]
        file_path = uploaded_file.path  

        # Create a new TextToSQL instance with the uploaded PDF
        text2sql = TextToSQL(file_path, "faiss_index")

        await cl.Message("✅ CSV uploaded successfully! Now ask your question.").send()
        return

    # If no PDF is uploaded yet, ask for one
    if text2sql is None:
        await cl.Message("⚠️ Please upload a CSV first.").send()
        return

    # Process user query after PDF is uploaded
    user_query = message.content.strip()
    sql_query = text2sql.generate_sql_query(user_query)
    
    await cl.Message(content=f"```sql\n{sql_query.content}\n```", author="SQL Generator").send()

if __name__ == "__main__":
    cl.run()



















# import chainlit as cl
# from aws_rag_txt2sql_csv import TextToSQL  # Ensure this file is named text_to_sql.py or modify accordingly

# # Initialize the TextToSQL class with the CSV file path and FAISS index storage path
# text2sql = TextToSQL("/home/sarveshharikant/EXIMIETAS/SARVESH/AWS/archive/Students data.csv", "faiss_index_csv")

# @cl.on_message
# async def handle_message(message: cl.Message):
#     """Handles user input and generates an SQL query based on the given natural language question."""
#     user_query = message.content.strip()
#     sql_query = text2sql.generate_sql_query(user_query)
    
#     await cl.Message(content=f"```sql\n{sql_query.content}\n```", author="SQL Generator").send()

# if __name__ == "__main__":
#     cl.run()
