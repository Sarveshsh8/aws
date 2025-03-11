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
    
    # Create message placeholders for streaming updates
    answer_msg = cl.Message(content="**Answer:** Processing...", author="SQL Generator")
    await answer_msg.send()
    
    sql_msg = cl.Message(content="```sql\nAnalyzing data...\n```", author="SQL Generator")
    await sql_msg.send()
    
    sql_ans_msg = cl.Message(content="**Output:** Waiting for results...", author="SQL Generator")
    await sql_ans_msg.send()
    
    # Start the stream
    full_response = ""
    current_json = {}
    
    # Get the streaming generator
    response_generator = text2sql.generate_sql_query_with_stream(user_query)
    
    # Process each chunk
    async for chunk, complete_response in response_generator:
        full_response = complete_response
        
        # Try to parse the current JSON (it might be incomplete)
        try:
            current_json = json.loads(full_response)
            
            # Update the message components as data comes in
            if 'Answer' in current_json:
                await answer_msg.update(content=f"**Answer:** {current_json['Answer']}")
            
            if 'SQL Query' in current_json:
                sql_query = current_json['SQL Query']
                if sql_query:
                    await sql_msg.update(content=f"```sql\n{sql_query}\n```")
                else:
                    await sql_msg.update(content="```\nNo SQL Query needed\n```")
            
            if 'SQL Query Answer' in current_json:
                sql_answer = current_json['SQL Query Answer']
                await sql_ans_msg.update(content=f"**Output:** {sql_answer}")
                
        except json.JSONDecodeError:
            # This is expected during streaming as we might get partial JSON
            pass
    
    # Final update with complete response
    try:
        final_json = json.loads(full_response)
        
        Answer = final_json.get('Answer', 'No answer provided')
        SQL = final_json.get('SQL Query', None)
        ans = final_json.get('SQL Query Answer', None)
        
        # Final updates to message components
        await answer_msg.update(content=f"**Answer:** {Answer}")
        
        if SQL:
            await sql_msg.update(content=f"```sql\n{SQL}\n```")
        else:
            await sql_msg.update(content="```\nNo SQL Query needed\n```")
        
        if ans:
            await sql_ans_msg.update(content=f"**Output:** {ans}")
        else:
            await sql_ans_msg.update(content="**Output:** No results")
        
        # Display the saved pie chart if available
        file_path = "dynamic_pie_chart.png"
        if os.path.exists(file_path):
            await cl.Message(
                content="📊 **Generated Pie Chart:**",
                elements=[cl.Image(path=file_path)]
            ).send()
    
    except json.JSONDecodeError:
        # Handle case where final response is not valid JSON
        await answer_msg.update(content="**Answer:** Error: Could not parse response as JSON")
        await sql_msg.update(content="```\nError in processing\n```")
        await sql_ans_msg.update(content="**Output:** Error in processing")

if __name__ == "__main__":
    cl.run()