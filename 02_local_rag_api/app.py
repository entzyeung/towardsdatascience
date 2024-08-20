from fastapi import FastAPI, HTTPException
from typing import Optional, List
from pydantic import BaseModel, Field
import yaml
import logging
from llama_index.llms.ollama import Ollama # it has to be current version, the different version throws error even loaded successfully.
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler  # Correct import
from rag import RAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_file = "config.yml"

with open(config_file, "r") as conf:
    config = yaml.safe_load(conf)

class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)

class Response(BaseModel):
    search_result: str
    source: str

# Initialize the CallbackManager with LlamaDebugHandler for debugging
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

# Initialize the Ollama model with a CallbackManager
llm = Ollama(model=config["llm_name"], url=config["llm_url"], callback_manager=callback_manager)

rag = RAG(config_file=config, llm=llm)
index = rag.qdrant_index()

app = FastAPI()

@app.get("/")
def root():
    return {"message": "medRxiv AI"}

prompt_eng = "You are only allowed to answer basing on the provided context. If a response cannot be formed strictly using the context, politely say you don’t have knowledge about that topic."

@app.post("/api/search", response_model=Response, status_code=200)
def search(query: Query):
    try:
        query_engine = index.as_query_engine(similarity_top_k=query.similarity_top_k, output=Response, response_mode="tree_summarize", verbose=True)
        
        response = query_engine.query(query.query + prompt_eng)
        
        response_object = Response(
            search_result=str(response).strip(), source=[response.metadata[k]["file_path"] for k in response.metadata.keys()][0]
        )
        return response_object
    except ValueError as ve:
        logger.error(f"ValueError occurred: {str(ve)}")
        raise HTTPException(status_code=500, detail="Internal Server Error: Missing callback manager")
    except Exception as e:
        logger.error(f"Error occurred while processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
