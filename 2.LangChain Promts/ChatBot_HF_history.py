from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
chat_history=[]
model=ChatHuggingFace(llm=llm)
while True:
    user_input=input("You: ")
    chat_history.append(user_input)
    if user_input=="Exit" or user_input=='exit':
        break
    result=model.invoke(chat_history)
    print("AI: ",result.content)
print(chat_history)