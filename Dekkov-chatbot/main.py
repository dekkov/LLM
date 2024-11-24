import asyncio
from browser_use import Agent, Controller
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete


# Persist browser state across agents
controller = Controller()

# Initialize browser agent
agent = Agent(
    task="Go to google.com and find the information about the new patch of Valorant",
    llm=ChatOpenAI(model="gpt-4o", timeout=25, stop=None),
    controller=controller)


async def main():
    max_steps = 2
    # Run the agent step by step
    for i in range(max_steps):
        print(f'\n📍 Step {i+1}')
        try:
            action, result = await agent.run()
            print('Action:', action)
            print('Result:', result)
        except:
            print("An exception occurred")
            continue

        

        if result.done:
            print('\n✅ Task completed successfully!')
            print('Extracted content:', result.extracted_content)
            
            # Save extracted content to a text file
            try:
                with open('hehe.txt', 'w') as file:
                    file.write(result.extracted_content)
                print("Extracted content has been saved to text.txt")
            except:
                print("FDSJFKDLSJFKDS ERROR")
            
            break

asyncio.run(main())
WORKING_DIR = "./dickens"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

with open("hehe.txt") as f:
    rag.insert(f.read())

#Perform naive search
print(rag.query("what is Supervised Fine-Tuning", param=QueryParam(mode="naive")))


# # Perform local search
# print(rag.query("what is Supervised Fine-Tuning", param=QueryParam(mode="local")))

# # Perform global search
# print(rag.query("what is Supervised Fine-Tuning", param=QueryParam(mode="global")))

# # Perform hybrid search
# print(rag.query("what is Supervised Fine-Tuning", param=QueryParam(mode="hybrid")))