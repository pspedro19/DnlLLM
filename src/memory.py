import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

class EnhancedVectorMemory:
    def __init__(self, memory_file):
        self.memory_file = memory_file
        self.memory = self.load_memory()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def load_memory(self):
        memory = []
        try:
            with open(self.memory_file, 'r') as file:
                memory = json.load(file)
        except FileNotFoundError:
            pass
        return memory

    def save_memory(self):
        with open(self.memory_file, 'w') as file:
            json.dump(self.memory, file, indent=4)

    def add_to_memory(self, question, answer):
        self.memory.append({"Pregunta": question, "Respuesta": answer})
        self.save_memory()

    def get_closest_memory(self, query):
        query_tokens = self.tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            query_embedding = self.model(**query_tokens).last_hidden_state.mean(dim=1)
        
        highest_similarity = -1
        closest = None
        for entry in self.memory:
            entry_tokens = self.tokenizer(entry["Pregunta"], return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                entry_embedding = self.model(**entry_tokens).last_hidden_state.mean(dim=1)
            similarity = cosine_similarity(query_embedding, entry_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                closest = entry
        return closest["Respuesta"] if closest else None
