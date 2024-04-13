from sentence_transformers import SentenceTransformer, util
import json
from pathlib import Path

class EnhancedVectorMemory:
    def __init__(self, memory_file):
        self.memory_file = Path(memory_file)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory = self.load_memory()

    def load_memory(self):
         if not self.memory_file.exists():
              print("Memory file does not exist, starting with an empty memory.")
              return {}

         with self.memory_file.open('r') as file:
              data = json.load(file)
              if not isinstance(data, list):
                  raise ValueError("Memory data should be a list of dictionaries.")

              memories = {}
              for idx, item in enumerate(data):
                 if not isinstance(item, dict) or 'Pregunta' not in item or 'Respuesta' not in item:
                        raise ValueError("Each item must be a dictionary with 'Pregunta' and 'Respuesta' keys.")
                 question = item['Pregunta'].replace("\n", " ")
                 answer = item['Respuesta'].replace("\n", " ")
                 memories[f"context_{idx+1}"] = f"{question} {answer}"

              return memories

        

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
