import json
from sentence_transformers import SentenceTransformer, util
import os

class VectorMemory:
    def __init__(self, memory_file):
        self.memory_file = memory_file
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory = self.load_memory()

    def load_memory(self):
        with open(self.memory_file, 'r') as file:
            data = json.load(file)
        return data

    def save_memory(self):
        with open(self.memory_file, 'w') as file:
            json.dump(self.memory, file, indent=4)

    def update_memory(self, key, value):
        self.memory[key] = value
        self.save_memory()

    def delete_from_memory(self, key):
        if key in self.memory:
            del self.memory[key]
            self.save_memory()

    def retrieve_relevant_info(self, query):
        query_embedding = self.model.encode(query)
        max_similarity = 0
        most_relevant_key = None
        for key, value in self.memory.items():
            value_embedding = self.model.encode(value)
            similarity = util.pytorch_cos_sim(query_embedding, value_embedding)
            if similarity > max_similarity:
                max_similarity = similarity.item()
                most_relevant_key = key
        return self.memory[most_relevant_key] if most_relevant_key else None
