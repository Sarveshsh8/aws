# README: CrewAI SQL Query and Text Summary Agents

## Overview

This project utilizes CrewAI to create two AI agents: one for generating SQL queries from natural language queries and another for summarizing text documents. The system integrates knowledge sources from CSV and PDF files while leveraging AWS Bedrock's models for natural language processing.

## Dependencies

Ensure you have the following dependencies installed before running the project:

```bash
pip install crewai pandas
```

## Configuration

The project is configured to use AWS Bedrock as the LLM provider. The model used is:

```python
model_id = "model_id"
```

Additionally, an embedding configuration is defined as:

```python
config = {
    "provider": "aws_bedrock",
    "config":{
      "model": "embedding model id",
      "vector_dimension": 1024
    }
}
```

## Agents and Tasks

### SQL Query Generation Agent

* **Role:** Text2SQL
* **Goal:** Convert natural language queries into SQL queries.
* **Task Steps:**
  1. Analyze user queries to understand intent.
  2. Identify database schema details (tables, columns, conditions).
  3. Construct an optimized SQL query.
  4. Optimize for efficiency.
  5. Validate for errors before returning.
* **Expected Output:** A syntactically correct SQL query.

### Text Summarization Agent

* **Role:** Text Summary
* **Goal:** Generate concise and accurate summaries of documents.
* **Task Steps:**
  1. Analyze the document.
  2. Extract key points.
  3. Structure the summary clearly.
  4. Ensure clarity and conciseness.
  5. Validate for accuracy.
* **Expected Output:** A well-structured and meaningful summary.

## Execution

The CrewAI framework orchestrates the execution of both agents using a sequential process.

```python
crew = Crew(
    agents=[sql_agent, sql_des_agent],
    tasks=[sql_plan, sql_des_plan],
    process=Process.sequential,
    max_rpm=10,
    cache=True,
    knowledge_sources=[csv_source, pdf_source],
    verbose=True,
)
```

### Running the Agents

To execute the workflow, initialize the input query and call the `kickoff` method:

```python
query = {"topic": "Fetch me the GPA of ID 1141"}
result = crew.kickoff(inputs=query)
```

The result can be retrieved using:

```python
agent_output = [i.raw for i in result.tasks_output]
print('agent_outputs', agent_output)
```

### How to Run the Code

To run the script, execute the following command in your terminal:

```bash
python agentic_aws.py
```

## File Structure

```
project_folder/
|-- data.csv  # CSV data source
|-- details.pdf  # PDF data source
|-- agentic_aws.py  # Python script for executing the agents
|-- README.md  # This document
```

## Future Enhancements

* Add more knowledge sources (e.g., database connections).
* Implement error handling and logging.
* Improve SQL optimization techniques.
* Extend summarization to handle multi-document processing.

## Author

Sarvesh

## License

This project is licensed under the MIT License.
