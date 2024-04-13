import asyncio
import torch
import json
import openai  # Import OpenAI library
from langchain.llms import HuggingFaceLLM
from langchain.chains import SingleTurnChain
from langchain.prompts import RolePlayingPrompt
from langchain.schema import Role
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import DataParallel

class Memory:
    def __init__(self, file_path):
        self.file_path = file_path
        try:
            with open(file_path, "r") as file:
                self.memory = json.load(file)
        except FileNotFoundError:
            self.memory = {}

    def get_closest_memory(self):
        # Retrieve the last context used
        return self.memory.get("latest_context", "")

    def update_memory(self, context):
        # Update memory with the latest context
        self.memory["latest_context"] = context
        with open(self.file_path, "w") as file:
            json.dump(self.memory, file)

# Define the role of the agent
dnl_agent_role = Role(
    name="DNL Agent",
    description="I am DNL Agent, your assistant here to help you with sales and customer service."
)

# Define a role-playing prompt with the agent role
prompt_builder = RolePlayingPrompt(
    roles=[dnl_agent_role],
    include_role_name_in_prompts=True
)

# Define the main function to handle the sales agent interaction
async def run_sales_agent(model, tokenizer, memory):
    # Initialize the Hugging Face language model with parallel processing
    hf_llm = HuggingFaceLLM(model, tokenizer=tokenizer)

    # Create a single-turn chain with the role-playing prompt
    chain = SingleTurnChain(llm=hf_llm, prompt_builder=prompt_builder)

    print("Welcome! You're chatting with DNL Agent. How may I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("DNL Agent: It was a pleasure assisting you. Goodbye!")
            break
        # Retrieve memory context
        memory_context = memory.get_closest_memory()
        enhanced_input = f"{memory_context} {user_input}" if memory_context else user_input
        response = await asyncio.create_task(chain.run_turn({"user": enhanced_input}))
        # Update memory with the latest interaction
        memory.update_memory(enhanced_input)
        print("DNL Agent:", response)

# Run the sales agent interaction loop
if __name__ == "__main__":
    openai_api_key = input("Please enter your OpenAI API key: ")  # Request OpenAI API key
    model_checkpoint_path = "/DnlLLM/src/results/20240412_211227/checkpoint-225"
    memory_path = "../data/memory.json"
    
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)

    # Utilize all available GPUs
    if torch.cuda.is_available():
        model.cuda()
        model = DataParallel(model)

    # Initialize memory system with the specified path
    memory = Memory(memory_path)
    
    asyncio.run(run_sales_agent(model, tokenizer, memory))
