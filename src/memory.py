from sentence_transformers import SentenceTransformer, util
import json
from pathlib import Path

class EnhancedVectorMemory:
    def __init__(self, memory_file):
        self.memory_file = Path(memory_file)
        self.memory = self.load_memory()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_memory(self):
        if not self.memory_file.exists():
            return {}
        with self.memory_file.open('r') as file:
            data = json.load(file)
            if not all(isinstance(item, dict) and 'Pregunta' in item and 'Respuesta' in item for item in data):
                raise ValueError("JSON data is not in the expected format: each item must be a dictionary with 'Pregunta' and 'Respuesta'")
            return {f"context_{idx+1}": f"{item['Pregunta']} {item['Respuesta']}"
                    for idx, item in enumerate(data)}

    def save_memory(self):
        with self.memory_file.open('w') as file:
            json.dump(self.memory, file, indent=4)

    def add_to_memory(self, key, value):
        self.memory[key] = value
        self.save_memory()

    def get_closest_memory(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        memories = list(self.memory.values())
        memory_embeddings = self.model.encode(memories, convert_to_tensor=True)
        distances = util.pytorch_cos_sim(query_embedding, memory_embeddings)[0]
        closest_idx = torch.argmax(distances).item()
        return memories[closest_idx]
