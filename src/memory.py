from sentence_transformers import SentenceTransformer, util
import json
from pathlib import Path

class EnhancedVectorMemory:
    def __init__(self, memory_file):
        self.memory_file = memory_file
        self.memory = self.load_memory()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_memory(self):
        memory = {}
        try:
            with open(self.memory_file, 'r') as file:
                memory = json.load(file)
        except FileNotFoundError:
            pass
        return memory

    def save_memory(self):
        with open(self.memory_file, 'w') as file:
            json.dump(self.memory, file, indent=4)

    def add_to_memory(self, key, value):
        self.memory.setdefault(key, []).append(value)
        self.save_memory()

    def get_closest_memory(self, query):
        query_embedding = self.model.encode(query)
        closest = None
        highest_similarity = -1
        for values in self.memory.values():
            for value in values:
                value_embedding = self.model.encode(value)
                similarity = util.pytorch_cos_sim(query_embedding, value_embedding)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    closest = value
        return closest

