# Ensure memory.py is accessible from this script, adjust the import path as needed
from memory import EnhancedVectorMemory

import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import DataParallel

class SimpleLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    async def generate_text(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        actual_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        output = actual_model.generate(
            input_ids,
            max_new_tokens=30,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=3,
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
        memory_response = memory.get_closest_memory(user_input)  # Removed 'await' here
        enhanced_input = f"{memory_response} {user_input}" if memory_response else user_input
        response = await llm.generate_text(enhanced_input)
        memory.add_to_memory(user_input, response)  # Assuming add_to_memory is not async
        print("DNL Agent:", response)

if __name__ == "__main__":
    tokenizer_checkpoint_path = "/DnlLLM/src/DnlModel/DnlModel"
    model_checkpoint_path = "/DnlLLM/src/DnlModel/checkpoint-225"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path)

    if torch.cuda.is_available():
        model = model.cuda()
        model = DataParallel(model)

    memory_path = "/DnlLLM/data/memory.json"
    memory = EnhancedVectorMemory(memory_path)
    llm = SimpleLLM(model, tokenizer)

    asyncio.run(run_sales_agent(llm, memory))
