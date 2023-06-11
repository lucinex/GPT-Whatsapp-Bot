import os
import uuid
from dotenv import load_dotenv
from twilio.rest import Client
from fastapi import FastAPI, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from src.chatbot.agent import Agent_BOT
from src.conf import CLIENT
load_dotenv()
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
my_number = os.environ["MY_NUMBER"]
twilio_number = os.environ["TWILIO_NUMBER"]

client = Client(account_sid, auth_token)
app = FastAPI()
agent = Agent_BOT(memory_client= CLIENT)


def send_message(body_text):
    client.messages.create(
        from_=twilio_number,
        body=body_text,
        to=my_number,
    )


def reply(text):
    return agent.run(text)


@app.post("/hook_chat")
async def chat(Body: str = Form(...)):
    message = str(Body)
    print(message)
    if message == "yessir":
        response = "Bulls EYE"
    else:
        response = reply(message)
    return send_message(response)
