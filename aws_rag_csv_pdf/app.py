import os
import chainlit as cl
from aws_rag_pdf import TextToSQL
import json  



# Global variables to store uploaded file paths
csv_path = None
pdf_path = None
text2sql = None  

@cl.on_message
async def handle_message(message: cl.Message):
    global csv_path, pdf_path, text2sql

    # Check if files are uploaded
    if message.elements:
        for uploaded_file in message.elements:
            file_path = uploaded_file.path  
            
            if file_path.endswith(".csv"):
                csv_path = file_path
            elif file_path.endswith(".pdf"):
                pdf_path = file_path

        if csv_path and pdf_path:
            text2sql = TextToSQL(csv_path, pdf_path, "faiss_index")
            await cl.Message("✅ CSV and PDF uploaded successfully! Now ask your question.").send()
        elif csv_path:
            await cl.Message("✅ CSV uploaded! Now upload a PDF.").send()
        elif pdf_path:
            await cl.Message("✅ PDF uploaded! Now upload a CSV.").send()
        return

    # Ensure both CSV and PDF are uploaded before processing queries
    if not csv_path or not pdf_path:
        await cl.Message("⚠️ Please upload both a CSV and a PDF first.").send()
        return

    # Process user query after both files are uploaded
    user_query = message.content.strip()
    response = text2sql.generate_sql_query(user_query)

    out = json.loads(response)
    Answer = out.get('Answer', 'No answer provided')  # Fetch the answer safely
    SQL = out.get('SQL Query', None)  # Fetch the SQL query safely
    ans = out.get('SQL Query Answer',None)


    # Send the answer separately
    await cl.Message(
        content=f"**Answer:** {Answer}",
        author="SQL Generator"
    ).send()

    # Send the SQL query separately
    await cl.Message(
        content=f"```sql\n{SQL if SQL else 'No SQL Query'}\n```",
        author="SQL Generator"
    ).send()

    #SQL ANSWER
    await cl.Message(
        content=f"**Output:** {ans}",
        author="SQL Generator"
    ).send()

    # ✅ **Display the already saved Pie Chart Image**
    file_path = "dynamic_pie_chart.png"

    if os.path.exists(file_path):
        await cl.Message(
            content="📊 **Generated Pie Chart:**",
            elements=[cl.Image(path=file_path)]
        ).send()
    else:
        print("Pie chart not found, skipping image display.")

if __name__ == "__main__":
    cl.run()





























































































# import os
# import chainlit as cl
# from aws_code_for_test import TextToSQL  

# # Global variable to store the TextToSQL instance
# text2sql = None  

# doc_paths = {
#     "csv": [],
#     "pdf": []
# }

# def reset_text2sql():
#     global text2sql
#     if doc_paths["csv"] or doc_paths["pdf"]:
#         text2sql = TextToSQL(
#             csv_paths=doc_paths["csv"], 
#             pdf_paths=doc_paths["pdf"], 
#             faiss_index_path="faiss_index"
#         )

# @cl.on_message
# async def handle_message(message: cl.Message):
#     global text2sql
    
#     # Handle file uploads
#     if message.elements:
#         for uploaded_file in message.elements:
#             file_path = uploaded_file.path  
            
#             if file_path.endswith(".csv"):
#                 doc_paths["csv"].append(file_path)
#             elif file_path.endswith(".pdf"):
#                 doc_paths["pdf"].append(file_path)
            
#         reset_text2sql()
#         await cl.Message("✅ Files uploaded successfully! Now ask your question.").send()
#         return

#     # Ensure files are uploaded before querying
#     if text2sql is None:
#         await cl.Message("⚠️ Please upload at least one CSV or PDF first.").send()
#         return

#     # Process user query
#     user_query = message.content.strip()
#     response = text2sql.generate_sql_query(user_query)
    
#     await cl.Message(content=f"**Answer:** {response.answer}\n\n```sql\n{response.sql_query or 'None'}\n```", author="SQL Generator").send()

# if __name__ == "__main__":
#     cl.run()
