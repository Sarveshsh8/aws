from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
import pandas as pd
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai import Agent, Task, Crew, Process, LLM
from crewai import Agent
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage



# CrewAI LLM config uses LiteLLM
model_name = "claude" #llama you can choose your llm
model_id = "arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n" ## you can pass your arn model id

llm = LLM(model=f"bedrock/{model_name}/{model_id}")


csv_source = CSVKnowledgeSource(
    file_paths=["data.csv"]
)

pdf_source = PDFKnowledgeSource(
    file_paths=["details.pdf"]
)

config = {
    "provider": "aws_bedrock",
    "config":{
      "model": "amazon.titan-embed-text-v2:0",
      "vector_dimension": 1024
    }
    }


## AGENT 1

sql_agent = Agent(
    llm=llm,
    role="Text2Sql",
    goal="To generate SQL queries based on natural language user queries, ensuring accuracy, efficiency, and relevance to the requested data retrieval task.",
    backstory="""You are a Text2SQL agent specializing in converting user queries written in natural language into optimized SQL queries.
                 Your query : {topic1}
                 Your role is to understand the user's intent, identify relevant database tables, columns, and conditions, and construct syntactically correct and efficient SQL statements. 
                 You ensure that the generated queries retrieve the precise data needed while following best practices for performance and security.""",
    allow_delegation=True,
    verbose=False,
    embedder = config,
)


sql_plan = Task(
    description=(
        "1. Analyze the user query to understand intent, "
            "identifying key entities such as tables, columns, "
            "conditions, and relationships.\n"
        "2. Map the extracted entities to the database schema, "
            "ensuring the correct table and field selection.\n"
        "3. Construct a syntactically correct SQL query that "
            "accurately retrieves the required information.\n"
        "4. Optimize the query for performance, avoiding redundancy "
            "and ensuring efficiency.\n"
        "5. Validate the generated SQL against possible errors or "
            "logical inconsistencies before returning it to the user."
    ),
    expected_output="Strictly provide me only the sql query code.",
    agent=sql_agent,
)


# AGENT 2 (text summary)

# %info
sql_des_agent = Agent(
    llm=llm,
    role="Text Summary",
    goal="To generate a concise and accurate summary of a given pdf document, capturing its key points while maintaining clarity and relevance.",
    backstory="""You are a Text Summary agent specializing in extracting key information from documents to generate concise and meaningful summaries.
                Your query : {topic2}
                 Your role is to analyze the content, identify essential details, and present a clear, structured summary while maintaining accuracy, relevance, and coherence.""",
    allow_delegation=True,
    verbose=False,
    embedder = config,
)


sql_des_plan = Task(
    description=(
        "1. Analyze the document to identify key information and main ideas.\n"
        "2. Extract essential details while maintaining clarity and relevance.\n"
        "3. Structure the summary to ensure coherence and readability.\n"
        "4. Optimize for conciseness without losing important context.\n"
        "5.Validate the summary to ensure accuracy and completeness before presenting it."
    ),
    expected_output="A concise and well-structured summary that accurately captures the key points of the document while ensuring clarity, relevance, and coherence.",
    agent=sql_des_agent,
)


# CREW AGENTIC CALL


crew = Crew(
    agents = [sql_agent,sql_des_agent],
    tasks = [sql_plan,sql_des_plan],
    process = Process.sequential,
    max_rpm =10,
    memory=True,
    cache=True,
    knowledge_sources=[csv_source,pdf_source],
    verbose=True,
)

query = {"topic1": "Fetch me the GPA of ID 1141 ","topic2":"give me the summary of the given document"}

result = crew.kickoff(inputs=query)

agent_output= [i.raw for i in result.tasks_output]

print('agent_outputs',agent_output)
