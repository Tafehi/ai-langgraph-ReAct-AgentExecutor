import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model


class OllamaLLM:
    def __init__(self):
        load_dotenv()
        self._model = os.getenv("OLLAMA_LLM")

    def get_llm(self):
        """Initialize and return the Ollama Embedding model."""
        try:
            if not self._model:
                raise ValueError("LLM_MODEL environment variable is not set.")

            return init_chat_model(model=self._model, model_provider="ollama")

                    # other params ...


        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama Embedding model: {e}")
