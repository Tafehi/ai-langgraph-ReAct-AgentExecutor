from typing import List, Union, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode
from tools import math_tool, serpApi
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from models.ollama_model import OllamaLLM
from datetime import datetime


class State(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], add_messages]
    message_type: str | None


class AgentExec:
    def __init__(self):
        self.state: State = {
            "messages": [],
            "message_type": None
        }
        load_dotenv()
        self.llm_provider = "ollama"
        self.llm_model = OllamaLLM.get_llm()

    def __agent_reason_template(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are helpful assistant.
                    Current time: {current_time}

                    Follow these steps:
                    1. Provide the best response for user input.
                    2. Use tools if necessary to achieve the best and update results.
                    3. Recommend 1-3 search queries to improve your answer.
                    4. Add date and time of when data is extracted.""",
                ),
                MessagesPlaceholder(variable_name="text"),
            ]
        )

    def agent_run(self):
        """ Run the agen reasoning node"""
        lst_msg = self.state['messages']
        response = self.llm_model.invoke(self.__agent_reason_template()+lst_msg)
        return {"messages": [response]}


    def tool_nodes(self):
        return ToolNode[math_tool, serpApi]