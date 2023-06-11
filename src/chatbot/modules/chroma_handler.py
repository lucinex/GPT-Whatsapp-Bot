import pickle
import os
from pathlib import Path

from chromadb.utils import embedding_functions
import chromadb
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    GPTVectorStoreIndex,
    LangchainEmbedding,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    download_loader,
    PromptHelper,
)
from langchain.agents import ZeroShotAgent, Tool

from llama_index.langchain_helpers.text_splitter import (
    TokenTextSplitter,
)
from llama_index.node_parser import SimpleNodeParser

from llama_index.indices.vector_store.retrievers import VectorIndexAutoRetriever
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores import ChromaVectorStore


from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore


from llama_index import QueryBundle
from llama_index import ResponseSynthesizer
from llama_index.query_engine import RetrieverQueryEngine

from llama_index.retrievers import VectorIndexRetriever
from llama_index.optimization import SentenceEmbeddingOptimizer

from src.chatbot.modules.dataloaders import IngestDocument
from langchain.chat_models import ChatOpenAI
import logging

logger = logging.basicConfig(level=logging.INFO)
logging.debug("Chroma handler .py ")


class ChromaStore:
    """
    to manage chroma collections and
    """

    def __init__(self, client):
        self.EMBED_FUNC = embedding_functions.SentenceTransformerEmbeddingFunction(
            device="cpu", model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.client = client

    def get_all_collections(self):  # name of all collections
        collections = self.client.list_collections()
        col_names = [col.name for col in collections]
        return col_names

    def get_collection(self, collection_name):
        return self.client.collection(collection_name)

    def get_or_create_collection(self, collection_name):
        return self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.EMBED_FUNC
        )

    def delete_ids_from_collection(self, collection_name, doc_ids):
        return self.get_or_create_collection(collection_name).delete(ids=doc_ids)


class ChromaHandler(IngestDocument):
    """
    Get files -> get doc -> save as keyword_index and faiss index with description
    description _path ->
    """

    EMBED_MODEL = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 32
    context_info = {}

    def __init__(self, save_dir, client, context="Research"):
        self.context = context
        self.save_dir = save_dir + "/" + self.context
        Path(self.save_dir).mkdir(exist_ok=True)
        self.context_info = self.load_context_from_json()
        self.cdb_store = ChromaStore(client)
        self.text_splitter = TokenTextSplitter(
            chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP
        )
        self.node_parser = SimpleNodeParser(
            text_splitter=self.text_splitter,
            include_extra_info=True,
            include_prev_next_rel=True,
        )
        self.llm_predictor_gpt = LLMPredictor(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4", max_tokens=1000)
        )
        self.service_ctx = ServiceContext.from_defaults(
            llm_predictor=self.llm_predictor_gpt,
            embed_model=self.EMBED_MODEL,
            node_parser=self.node_parser,
        )

    def load_context_from_json(self):  # save context dict
        logging.info(f"loading {self.context}_context.json")
        fp = self.save_dir + f"/{self.context}_context.json"
        try:
            if os.path.isfile(fp):
                with open(fp, "rb") as ff:
                    files = pickle.load(ff)
                return files
            else:
                return {}
        except Exception as e:
            print(e)
            raise RuntimeError(
                "Invalid context.json , something went wrong while loading"
            )

    def save_context_to_json(self):  # load context dict
        logging.info(f"saving {self.context}_context.json")
        fp = self.save_dir + f"/{self.context}_context.json"
        try:
            if self.context_info != {}:
                with open(fp, "wb") as ff:
                    pickle.dump(self.context_info, ff)
                return
            else:
                return
        except Exception as e:
            print(e)
            raise RuntimeError(
                "Invalid context.json , something went wrong while loading"
            )

    def create_llama_nodes(self, document):
        # print(dir(self.node_parser))
        nodes = self.node_parser.get_nodes_from_documents(document)
        new_nodes = []
        for e, n in enumerate(nodes):
            nu_node = nodes[e]
            nu_node.extra_info = {**nu_node.extra_info, "page_no": str(e + 1)}
            new_nodes.append(nu_node)
        return new_nodes

    def save_to_store(self, nodes):
        vector_store = ChromaVectorStore(
            chroma_collection=self.cdb_store.get_or_create_collection(self.context)
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = GPTVectorStoreIndex(
            nodes, service_context=self.service_ctx, storage_context=storage_context
        )
        return index

    def persist_document(self, filepath):
        if Path(filepath).is_file() and not self.check_if_filename_already_indexed(
            filepath
        ):
            logging.info(f"Saving to collection {self.context}")
            context_info = {}
            filename = "".join(filepath.split("/")[-1].split(".")[:-1])
            logging.info(f"Saving file {filename}")
            save_path = self.save_dir
            document = self.load_document(
                filepath, extra_info={"context_filepath": self.save_dir}
            )
            nodes = self.create_llama_nodes(document)
            document_id = nodes[0].ref_doc_id
            doc_ids = [n.doc_id for n in nodes]
            doc_size = len(nodes)
            context_info = {
                document_id: {
                    "doc_ids": doc_ids,
                    "filename": filename,
                    "filepath": filepath,
                    "doc_size": doc_size,
                }
            }

            self.save_to_store(nodes)
            logging.info(f"Saved to chroma {filename}")
            self.context_info.update(context_info)

            if self.cdb_store.client.persist():
                self.save_context_to_json()
                logging.info(f"Saved {filename}")
                return f"Saved {filename}"
            else:
                logging.error(
                    f"Error while writing to chrome client , collection {self.context}, file: {filename}"
                )
                return f"Error while writing to chrome client , collection {self.context}, file: {filename}"

        elif self.check_if_filename_already_indexed(filepath):
            logging.info(
                f" File with filename already exists. {filepath}. Will not do anything"
            )
            return (
                f" File with filename already exists. {filepath}. Will not do anything"
            )

    def get_all_indexed_files(self):
        if self.context_info != {}:
            return_vals = []
            for i, j in self.context_info.items():
                filename = j.get("filename", "")
                file_path = j.get("filepath", "")
                doc_id = i
                return_vals.append((filename, file_path, doc_id))
            return return_vals
        else:
            return []

    def get_doc_id_for_filepath(self, filepath):
        if self.context_info != {}:
            return_vals = ""
            for i, j in self.context_info.items():
                if j.get("filepath", "") == filepath:
                    return i
                else:
                    continue
            return return_vals
        else:
            raise NotImplementedError("No context info present")

    def check_if_filename_already_indexed(self, filepath):
        if os.path.isfile(filepath):
            filenames = self.get_all_indexed_files()
            filename = [i for i, _, _ in filenames if i == filepath.split("/")[-1]]
            if len(filename) > 0:
                return True
            return False
        else:
            raise FileNotFoundError(f"No file exists as {filepath}")

    def delete_document_by_name(self, filename: str):
        all_filenames_indexed = [(i, j) for i, _, j in self.get_all_indexed_files()]
        for i, j in all_filenames_indexed:
            if i == filename:
                logging.info(f"Found {filename}, in context {self.context} deleting.")

                # index = self.save_to_store([])
                # index.delete_ref_doc(j)
                doc_ids = self.context_info[j]["doc_ids"]
                self.cdb_store.delete_ids_from_collection(self.context, doc_ids)
                self.context_info.pop(j)
                if self.cdb_store.client.persist():
                    self.save_context_to_json()
                    logging.info(f"Successfully deleted file: {filename}.")
                else:
                    logging.error(
                        f"Could not persist after deleltion for context: {self.context} which has file: {filename}."
                    )
                return
            else:
                continue
        logging.info(f"File {filename} is not indexed in context {self.context}.")


class ChromaTools(ChromaHandler):
    def __init__(self, save_dir, client, context):
        ChromaHandler.__init__(self, save_dir=save_dir, client=client, context=context)

    def get_retriver(self, filters={}):
        index = self.save_to_store([])
        if filters != {}:
            apply_filters = []
            for f, p in filters.items():
                apply_filters.append(ExactMatchFilter(key=f, value=p))

            filters = MetadataFilters(filters=apply_filters)
            retriever = index.as_retriever(filters=filters)

        else:
            retriever = index.as_retriever()
        return retriever

    def get_query_engine(self, similarity_top_k=2, verbose=True):
        index = self.save_to_store([])
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
        )
        response_synthesizer = ResponseSynthesizer.from_args(
            service_context=self.service_ctx,
            # text_qa_template=qa_chat_prompt,
            response_mode="compact",
            # optimizer=SentenceEmbeddingOptimizer(
            #     embed_model = self.EMBED_MODEL
            #     percentile_cutoff=0.5
            # ),
            # node_postprocessors=[
            #     SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)
            # ],
        )

        q_engine = RetrieverQueryEngine.from_args(
            retriever=retriever, response_synthesizer=response_synthesizer
        )
        return q_engine

    def get_langchain_tool(self, similarity_top_k=2, verbose=True):  # query all
        name = f"Query_{self.context}_Documents"
        q_engine = self.get_query_engine(
            similarity_top_k=similarity_top_k, verbose=verbose
        )

        def func(x):
            # raise Exception("Test , tool executed to check callbacks")
            return q_engine.query(x)

        filenames = [i for i, _, _ in self.get_all_indexed_files()]
        description = (
            f"Retrive Information from Documents in context of '{self.context}'. Tool expects a proper question. Files indexed inside the database are : ["
            + " ,".join(filenames)
            + "]. Use this tool to ask questions pertaining to the indexed documents only"
        )
        tool = Tool(name=name, func=func, description=description)
        return tool

    def get_file_deletion_tool(self):
        name = f"Delete {self.context} Document"
        all_files = [i for i, _, _ in self.get_all_indexed_files()]
        description = (
            f"Delete Indexed Documents from context {self.context}. Input to the tool should one of these filenames: "
            + "["
            + " ,".join(all_files)
            + "]. "
        )

        def delete(filename_query):
            all_filenames = [i for i, _, _ in self.get_all_indexed_files()]
            if filename_query in all_filenames:
                self.delete_document_by_name(filename_query)
                return f"Successfully handled deletion of file: {filename_query} from context: {self.context}"

        tool = Tool(name=name, func=delete, description=description)
        return tool

    def get_file_addition_tool(self):
        name = f"Add {self.context} Document"
        all_files = [i for i, _, _ in self.get_all_indexed_files()]
        description = f"Add a new document to {self.context} Documents. Input should be a valid filepath."

        def add(filepath_query):
            all_filepaths = [i for _, i, _ in self.get_all_indexed_files()]
            if filepath_query not in all_filepaths:
                return self.persist_document(filepath_query)
                # return f"Successfully handled deletion of file: {filepath_query} from context: {self.context}"
            else:
                return f"File with filepath {filepath_query} already exists"

        tool = Tool(name=name, func=add, description=description)
        return tool


def ingest_document_by_context(filepath: str, context: str):
    from src.conf import CLIENT, AGENT_CHROMA_DATA_PATH

    client = CLIENT

    save_dir = AGENT_CHROMA_DATA_PATH

    chroma_handler = ChromaHandler(save_dir, client, context)
    chroma_handler.persist_document(filepath)
    print(chroma_handler.get_all_indexed_files())
    return
