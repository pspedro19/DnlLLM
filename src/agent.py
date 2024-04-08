import os
from langchain_community import LangChainAgent
from langchain.prompts import Prompt
from memory import VectorMemory
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define paths
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Load structured data from JSON file
memory_file = os.path.join(data_dir, 'memory.json')
memory = VectorMemory(memory_file)

# Define the path to the locally saved model
model_path = "../mistral_7b_guanaco"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Initialize LangChain agent with the fine-tuned model
llm = HuggingFace(model=model, tokenizer=tokenizer)
agent = LangChainAgent(llm=llm)

# Define prompt for sales interaction
sales_prompt = Prompt(
    template="Hello, I am an intelligent sales assistant. How can I help you today?",
    stop_sequences=["\n"]
)

# Main interaction loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Retrieve relevant information from memory
    relevant_info = memory.retrieve_relevant_info(user_input)

    # Generate response using LangChain agent and memory
    response = agent.generate(
        prompt=sales_prompt.fill(user_input=user_input, memory_info=relevant_info),
        max_length=50
    )

    print("Assistant:", response)
