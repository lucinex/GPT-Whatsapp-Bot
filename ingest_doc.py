from dotenv import load_dotenv
from src.chatbot.modules.chroma_handler import (
    ingest_document,
    ChromaHandler,
    ChromaStore,
)
from src.conf import CLIENT

if __name__ == "__main__":
    load_dotenv()
    file_p = "/home/srdevl191/Desktop/Whatsapp_BOT/src/data/Documents/Thesis_Preparation_Submission_Guidelines_PhD_PG.pdf"
    file_p2 = "/home/srdevl191/Desktop/Whatsapp_BOT/src/data/Documents/DT20223560376_Offer_Letter(TCS_R&I).pdf"
    context = "test"

    client = CLIENT
    # CLIENT.reset()
    # CLIENT.persist()

    print(client.list_collections())
    col_name = "test"
    # col_name = ChromaStore(client).get_all_collections()[0]
    # collection = client.get_collection(col_name)

    ingest_document(file_p2, context)
    collection = client.get_collection("test")
    print(collection.count())

    ch = ChromaHandler("src/chatbot/agent_data", client, col_name)
    ret = ch.get_retriver()
    print(ret.retrieve("What is the salary?"))
