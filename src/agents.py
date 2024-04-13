import asyncio
import torch
import json
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
        return self.memory.get("latest_context", "")

    def update_memory(self, context):
        self.memory["latest_context"] = context
        with open(self.file_path, "w") as file:
            json.dump(self.memory, file)

class SimpleLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    async def generate_text(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
        output = self.model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.eos_token_id,
            early_stopping=True
        )
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output

async def run_sales_agent(llm, memory):
    print("Welcome! You're chatting with DNL Agent. How may I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("DNL Agent: It was a pleasure assisting you. Goodbye!")
            break
        memory_context = memory.get_closest_memory()
        enhanced_input = f"{memory_context} {user_input}" if memory_context else user_input
        response = await llm.generate_text(enhanced_input)
        memory.update_memory(enhanced_input)
        print("DNL Agent:", response)

if __name__ == "__main__":
    openai_api_key = input("Please enter your OpenAI API key: ")
    model_checkpoint_path = "/DnlLLM/src/results/20240412_211227/checkpoint-225"
    memory_path = "../data/memory.json"

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)

    if torch.cuda.is_available():
        model.cuda()
        model = DataParallel(model)

    memory = Memory(memory_path)
    llm = SimpleLLM(model, tokenizer)

    asyncio.run(run_sales_agent(llm, memory))

