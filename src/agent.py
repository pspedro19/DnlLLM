import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from memory import EnhancedVectorMemory
from langchain.llms import OpenAI, LLMChain
from langchain.prompts import PromptTemplate

class SalesAgent:
    def __init__(self, model_path, openai_api_key, memory_file="memory.json", max_execution_time=10):
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.openai_model = OpenAI(api_key=openai_api_key)

        self.memory = EnhancedVectorMemory(memory_file)

        self.prompt_template = PromptTemplate(
            template="You are a sales agent. Use past interactions and this context to enhance customer satisfaction: {input}."
        )

        self.max_execution_time = max_execution_time

        self.hf_agent = LLMChain(llm=self.hf_model, tokenizer=self.tokenizer)
        self.openai_agent = LLMChain(llm=self.openai_model, prompt=self.prompt_template)

    async def handle_query(self, user_input):
        memory_context = self.memory.get_closest_memory(user_input)
        enhanced_input = f"{user_input} {memory_context}" if memory_context else user_input
        
        response_hf = await asyncio.wait_for(self.hf_agent.run(enhanced_input), timeout=self.max_execution_time)
        response_openai = await asyncio.wait_for(self.openai_agent.run({"input": enhanced_input}), timeout=self.max_execution_time)
        
        final_response = response_openai if len(response_openai) < len(response_hf) else response_hf
        
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
    model_path = "../model/mistral_7b_guanaco"
    memory_path = "../data/memory.json"
    openai_api_key = "your-api-key"
    agent = SalesAgent(model_path, openai_api_key, memory_file=memory_path)
    asyncio.run(agent.run())
