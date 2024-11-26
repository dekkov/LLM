from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
app = FastAPI(
	title="LangChain Server",
	version="1.0",
	description="A simple API server using Langchain's Runnable interfaces",
)
# Define a route for the OpenAI chat model
add_routes(
	app,
	ChatOpenAI(),
	path="/openai",
)
# Define a route with a custom prompt
summarize_prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
add_routes(
	app,
	summarize_prompt | ChatOpenAI(),
	path="/summarize",
)

joke_prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
add_routes(
            app,
            joke_prompt | ChatOpenAI(),
            path="/joke",
)
if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="localhost", port=8000)