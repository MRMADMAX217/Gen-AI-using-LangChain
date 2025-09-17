from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

modell=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="conversational"
)
model=ChatHuggingFace(llm=modell)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)