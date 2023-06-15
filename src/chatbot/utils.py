import tiktoken
import os
from typing import List
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.langchain import LangchainEmbedding
from src.conf import EMBEDDING_MODEL_NAME
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def mkdir_if_dne(path):
    if os.path.isdir(path):
        return False
    else:
        os.mkdir(path)
        print(f"dir created : {path}")
        return True


def get_folders(path, exclude: list = None):
    folders = []
    if os.path.isdir(path):
        for i in os.listdir(path):
            if exclude:
                if i in exclude:
                    continue
            nw_path = path + "/" + i
            if os.path.isdir(nw_path):
                folders.append(nw_path)
    return folders


def get_embedding_function(model_name=EMBEDDING_MODEL_NAME, langchain=False):
    if langchain:
        EMBED_MODEL = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        )

    else:
        EMBED_MODEL = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    return EMBED_MODEL


class AgentSettings:
    PERSONA = ""
    AGENT_TYPE = "general"
    AGENT_KWARGS = {"kwargs": None}
    SAVE_DIR = "/src/chatbot/agent_settings.yaml"

    def __init__(self, dir):
        self.save_dir = dir

        pass

    # def set(self,**kwargs):

    #     pass
    # def refresh(self):
    #     # load json
    # def save(self):
    #     # save json
    # def set_default(self):
