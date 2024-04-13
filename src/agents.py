import asyncio
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import DataParallel
from sentence_transformers import SentenceTransformer, util

class Memory:
    def __init__(self, file_path, vector_db_path):
        self.file_path = file_path
        self.vector_db = SentenceTransformer(vector_db_path)
        try:
            with open(file_path, "r") as file:
                self.memory = json.load(file)
        except FileNotFoundError:
            self.memory = {}

    def get_closest_memory(self, query):
        if not self.memory:
            return ""
        query_embedding = self.vector_db.encode(query, convert_to_tensor=True)
        memories = {k: v for k, v in self.memory.items() if k.startswith("context_")}
        memory_embeddings = self.vector_db.encode(list(memories.values()), convert_to_tensor=True)
        distances = util.pytorch_cos_sim(query_embedding, memory_embeddings)[0]
        closest_idx = torch.argmax(distances).item()
        return list(memories.values())[closest_idx]

    def update_memory(self, context):
        context_id = f"context_{len(self.memory)}"
        self.memory[context_id] = context
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

        # Check if the model is wrapped with DataParallel and access the underlying model
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
        memory.update_memory(enhanced_input)
        print("DNL Agent:", response)

if __name__ == "__main__":
    model_checkpoint_path = "/DnlLLM/src/DnlModel/DnlModel"  # Ensure this points to the correct model directory
    memory_path = "/DnlLLM/data/memory.json"  # Corrected path to the memory file
    vector_db_path = "sentence-transformers/all-MiniLM-L6-v2"  # Confirm this path if it's a local directory or a model identifier from Hugging Face

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path)

    if torch.cuda.is_available():
        model = model.cuda()
        model = DataParallel(model)

    memory = Memory(memory_path, vector_db_path)
    llm = SimpleLLM(model, tokenizer)

    asyncio.run(run_sales_agent(llm, memory))

