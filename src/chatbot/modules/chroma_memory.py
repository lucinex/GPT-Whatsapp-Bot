from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from langchain.memory.chat_memory import BaseMemory
from langchain.memory.utils import get_prompt_input_key
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.memory import ReadOnlySharedMemory
from langchain.schema import get_buffer_string
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

# from src.conf import EMBEDDING_MODEL_NAME

from langchain.vectorstores import Chroma
from src.chatbot.utils import get_embedding_function


class CustomVectorStoreRetrieverMemory(BaseMemory):
    retriever: VectorStoreRetriever = Field(exclude=True)
    memory_key: str = "chat_history"  #: :meta private:
    ai_prefix = "assistant"
    human_prefix = "human"
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    chat_memory = ChatMessageHistory()
    return_docs: bool = False
    return_messages = True
    sep = "\n\t"
    """Whether or not to return the result of querying the database directly."""

    @property
    def memory_variables(self) -> List[str]:
        """The list of keys emitted from the load_memory_variables method."""
        return [self.memory_key]

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """Get the input key for the prompt."""
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def load_memory_variables(self, inputs: Dict[str, Any]):
        """Return history buffer."""
        self.chat_memory.clear()
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self.retriever.get_relevant_documents(query)
        self._add_chat_messages(docs)
        # result: Union[List[Document], str]
        if self.return_messages:
            return {self.memory_key: self.chat_memory.messages}
        else:
            return {self.memory_key: get_buffer_string(self.chat_memory.messages)}

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """Format context from this conversation to buffer."""
        # Each document should only include the current turn, not the chat history
        filtered_inputs = {k: v for k, v in inputs.items() if k != self.memory_key}
        texts = [
            f"{k}: {v}"
            for k, v in list(filtered_inputs.items()) + list(outputs.items())
        ]
        page_content = self.sep.join(texts)
        return [Document(page_content=page_content)]

    def _add_chat_messages(self, documents: Union[List[Document], str]):
        if type(documents) == str or (type(documents) == list and len(documents) == 0):
            return [{"role": self.ai_prefix, "content": "No previous conversations"}]
        else:
            messages = []
            for doc in documents:
                chunks = doc.page_content.split(self.sep)
                for chunk in chunks:
                    full_text, role = (
                        ":".join(chunk.split(":")[1:]),
                        chunk.split(":")[0],
                    )
                    if role == self.ai_prefix:
                        self.chat_memory.add_ai_message(full_text)
                    elif role == self.human_prefix:
                        self.chat_memory.add_human_message(full_text)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        new_output = {"output": outputs["output"]}
        documents = self._form_documents(inputs, new_output)
        self.retriever.add_documents(documents)

    def clear(self) -> None:
        """Nothing to clear."""
        self.chat_memory.clear()


def get_chroma_memory(client, k=1, read_only=False):
    embed_func = get_embedding_function()
    chromaVectorstore = Chroma(
        collection_name="Memory",
        embedding_function=embed_func,
        client=client,
    )
    retriver = chromaVectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    memory = CustomVectorStoreRetrieverMemory(retriever=retriver)
    if read_only:
        return ReadOnlySharedMemory(memory=memory)
    else:
        return memory
