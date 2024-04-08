import os
import asyncio
from langchain_community import LangChainAgent, HuggingFace  # Check if LangChainAgent can be imported
from langchain.prompts import Prompt
from memory import VectorMemory
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.agents import tool, AgentExecutor, format_to_openai_tool_messages, AgentActionMessageLog, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List  # Added import for List
import json

# Define a response schema
class Response(BaseModel):
    """Final response to the question being asked"""
    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(description="List of page chunks that contain answer to the question")

# Define custom parsing logic
def parse(output):
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)
    
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])
    
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    else:
        return AgentActionMessageLog(tool=name, tool_input=inputs, log="", message_log=[output])

# Define a simple tool for demonstration purposes
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

class SalesAgent:
    def __init__(self, max_execution_time=10):
        # Define paths
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

        # Load structured data from JSON file
        memory_file = os.path.join(self.data_dir, 'memory.json')
        self.memory = VectorMemory(memory_file)

        # Define the path to the locally saved model
        model_path = "../mistral_7b_guanaco"

        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Initialize LangChain agent with the fine-tuned model
        llm = HuggingFace(model=self.model, tokenizer=self.tokenizer)
        self.agent = LangChainAgent(llm=llm)

        # Define tools and bind them to the LLM
        self.tools = [get_word_length]
        llm_with_tools = llm.bind_functions(self.tools + [Response])

        # Define prompt for sales interaction with memory
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a very powerful sales assistant."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent={
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
                "chat_history": lambda x: x["chat_history"],
            } | self.prompt | llm_with_tools | parse,
            tools=self.tools,
            verbose=True,
            max_execution_time=max_execution_time
        )

        # Initialize chat history
        self.chat_history = []

    async def run(self):
        # Main interaction loop
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            # Update chat history
            self.chat_history.append(HumanMessage(content=user_input))

            # Generate response using the agent executor
            async for chunk in self.agent_executor.astream(
                {"input": user_input, "chat_history": self.chat_history}
            ):
                if "output" in chunk:
                    response = chunk["output"]
                    self.chat_history.append(AIMessage(content=response))
                    print("Assistant:", response)

if __name__ == "__main__":
    sales_agent = SalesAgent(max_execution_time=10)  # Set the timeout to 10 seconds
    asyncio.run(sales_agent.run())
