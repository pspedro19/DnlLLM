# memory.py needs to be accessible from this script, ensure it's in the same directory or adjust the import path
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
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        actual_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        output = actual_model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=3,  # Using more than one beam
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
        memory_context = memory.get_closest_memory(user_input)
        enhanced_input = f"{memory_context} {user_input}" if memory_context else user_input
        response = await llm.generate_text(enhanced_input)
        memory.add_to_memory(f"context_{len(memory.memory) + 1}", enhanced_input)  # Using add_to_memory method
        print("DNL Agent:", response)

if __name__ == "__main__":
    model_checkpoint_path = "/DnlLLM/src/DnlModel/DnlModel"
    memory_path = "/DnlLLM/data/memory.json"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path)

    if torch.cuda.is_available():
        model = model.cuda()
        model = DataParallel(model)

    memory = EnhancedVectorMemory(memory_path)
    llm = SimpleLLM(model, tokenizer)

    asyncio.run(run_sales_agent(llm, memory))
