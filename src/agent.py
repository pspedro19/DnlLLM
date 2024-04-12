import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure necessary libraries are installed
try:
    from sentence_transformers import SentenceTransformer  # Checks if sentence_transformers is installed
except ImportError:
    print("sentence_transformers is not installed. Install it using: pip install sentence_transformers")

try:
    from langchain_community.llms import OpenAI  # Updated imports
    from langchain_community.chains import LLMChain
    from langchain_community.prompts import PromptTemplate
except ImportError:
    print("langchain-community is not installed. Install it using: pip install -U langchain-community")

# Custom module, implement error handling inside the module or here if the module does not exist.
try:
    from memory import EnhancedVectorMemory
except ImportError as e:
    print(f"Failed to import custom memory module: {e}")
    print("Ensure your custom 'memory' module is correctly implemented and accessible.")

class SalesAgent:
    def __init__(self, model_path, adapter_path, openai_api_key, memory_file="memory.json", max_execution_time=10):
        # Load the model including its adapter
        try:
            self.hf_model = AutoModelForCausalLM.from_pretrained(model_path)
            self.hf_model.load_adapter(adapter_path)  # Load the adapter
        except Exception as e:
            print(f"Error loading model or adapter: {e}")
            return
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.openai_model = OpenAI(api_key=openai_api_key)
        self.memory = EnhancedVectorMemory(memory_file) if 'EnhancedVectorMemory' in globals() else None
        self.prompt_template = PromptTemplate(
            template="You are a sales agent. Use past interactions and this context to enhance customer satisfaction: {input}."
        )
        self.max_execution_time = max_execution_time
        self.hf_agent = LLMChain(llm=self.hf_model, tokenizer=self.tokenizer)
        self.openai_agent = LLMChain(llm=self.openai_model, prompt=self.prompt_template)

    async def handle_query(self, user_input):
        memory_context = self.memory.get_closest_memory(user_input) if self.memory else ""
        enhanced_input = f"{user_input} {memory_context}" if memory_context else user_input
        
        response_hf = await asyncio.wait_for(self.hf_agent.run(enhanced_input), timeout=self.max_execution_time)
        response_openai = await asyncio.wait_for(self.openai_agent.run({"input": enhanced_input}), timeout=self.max_execution_time)
        
        final_response = response_openai if len(response_openai) < len(response_hf) else response_hf
        if self.memory:
            self.memory.add_to_memory("conversation", {"query": user_input, "response": final_response})
        
        return final_response

    async def run(self):
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = await self.handle_query(user_input)
            print("Assistant:", response)

if __name__ == "__main__":
    model_checkpoint_path = "/DnlLLM/src/results/20240412_211227/checkpoint-200"
    adapter_path = "/DnlLLM/src/mistral_7b_guanaco/adapter"
    memory_path = "../data/memory.json"
    openai_api_key = "your-api-key"
    
    # Create the SalesAgent instance and start it
    agent = SalesAgent(model_checkpoint_path, adapter_path, openai_api_key, memory_file=memory_path)
    asyncio.run(agent.run())
