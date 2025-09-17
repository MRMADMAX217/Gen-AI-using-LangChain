# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# model = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#     task="conversational"
# )

# messages=[
#     SystemMessage(content='You are a helpful assistant'),
#     HumanMessage(content='Tell me about LangChain')
# ]

# result = model.invoke(messages)

# messages.append(AIMessage(content=result.content))

# print(messages)
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize the Hugging Face model
hf_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="conversational"
)
model = ChatHuggingFace(llm=hf_endpoint)
# Define your messages
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Hello, how are you?")
]

# Invoke the model
result = model.invoke(messages)

print(result)
