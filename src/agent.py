import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

# Definición de la clase SimpleLLM para simplificar la generación de texto con el modelo
class SimpleLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    async def run(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt")["input_ids"]
        output = self.model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output

# Definición de la clase SalesAgent para manejar las interacciones con el usuario
class SalesAgent:
    def __init__(self, model_checkpoint_path, openai_api_key, memory_file="memory.json", max_execution_time=10):
        # Carga del modelo y el tokenizer
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
        self.simple_llm = SimpleLLM(self.hf_model, self.tokenizer)
        self.openai_model = OpenAI(api_key=openai_api_key)
        self.memory = EnhancedVectorMemory(memory_file) if 'EnhancedVectorMemory' in globals() else None
        self.max_execution_time = max_execution_time

    async def handle_query(self, user_input):
        # Obtención de contexto de la memoria si está disponible
        memory_context = self.memory.get_closest_memory(user_input) if self.memory else ""
        enhanced_input = f"{user_input} {memory_context}" if memory_context else user_input

        # Generación de respuestas usando el modelo HF y el modelo OpenAI
        response_hf = await asyncio.wait_for(self.simple_llm.run(enhanced_input), timeout=self.max_execution_time)
        response_openai = await asyncio.wait_for(self.openai_model.run({"input": enhanced_input}), timeout=self.max_execution_time)

        # Selección de la respuesta más corta como final
        final_response = response_openai if len(response_openai) < len(response_hf) else response_hf
        if self.memory:
            self.memory.add_to_memory("conversation", {"query": user_input, "response": final_response})

        return final_response

    async def run(self):
        # Bucle principal para interactuar con el usuario
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = await self.handle_query(user_input)
            print("Assistant:", response)

if __name__ == "__main__":
    # Rutas de los archivos y parámetros necesarios
    model_checkpoint_path = "/DnlLLM/src/results/20240412_211227/checkpoint-225"
    memory_path = "../data/memory.json"
    openai_api_key = "your-api-key"

    # Creación de una instancia de SalesAgent y ejecución del bucle principal
    agent = SalesAgent(model_checkpoint_path, openai_api_key, memory_file=memory_path)
    asyncio.run(agent.run())
