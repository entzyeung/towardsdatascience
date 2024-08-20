### Loading the embedder

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.service_context import ServiceContext

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import qdrant_client
import yaml
########## 
# from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)



class RAG:
    def __init__(self, config_file, llm):
        self.config = config_file
        self.qdrant_client = qdrant_client.QdrantClient(
            url=self.config['qdrant_url']
        )
        self.llm = llm  # ollama llm
    
    def load_embedder(self):
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=self.config['embedding_model'])
        )
        return embed_model


    def qdrant_index(self):
        client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=self.config['collection_name']
        )
        # service_context = ServiceContext.from_defaults(
        #     llm=self.llm, embed_model="local:BAAI/bge-small-en-v1.5"
        # )
        ############################
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        # callback_manager = CallbackManager()
        service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=self.load_embedder(), chunk_size=self.config["chunk_size"], callback_manager=callback_manager
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store=qdrant_vector_store, service_context=service_context
        )
        return index

    # def _load_config(self, config_file):
    #     with open(config_file, "r") as stream:
    #         try:
    #             self.config = yaml.safe_load(stream)
    #         except yaml.YAMLError as e:
    #             print(e)

