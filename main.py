from dotenv import load_dotenv
import getpass
import os

from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()



if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")




template = """Question: {question}
Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="qwen3:8b")

chain = prompt | model

chain.invoke({"question": "What is LangChain?"})

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

#
# def main():
#     print("Hello from langchain-demo!")
#
# if __name__ == "__main__":
#     main()
