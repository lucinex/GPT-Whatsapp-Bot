# GPT_Whatsapp_BOT

This is minimal MVP of a Whatsapp assistant designed to help you ask mundane stuff. It also has some form of memory but still not very intensive at the moment. You can get things done in coherent manner nonetheless.

Designed with langchain and llama-index, powered by OpenAI and ChormaDB.

It Has Access to few of these tools:

1. DuckDuckGO Search : Search about current events.
2. Python Repl : Do Pythonic stuff !
3. Add / Delete Documents to vector db: You can ask which files you have access to and ask to specifically add or delete from the vector db. Should load a document before using the next tool.
4. Query_documents: Query the documents you've loaded. You can ask what documents you can query from.
5. Chat with CSV : Lets you chat with an agent that can access a CSV in the CSV Directory. You can choose which csvs to choose by asking the agent what csvs it has. Will load up a new agent with the mentioned csvs loaded.

Notes:
There are general keywords that can be sent via chat to change settings. Like

1. 'Analytics' : Show / Unshow Analytics, token consumption.
2. 'Reset' : Reset agent type ( should be used to give back command to general agent)
3. "Yessir": Sanity check . Should return a message saying "BullyEYE"

Also, Please change the name of the bot if you are using it. Default name is pretty bad.

## Installation

1. Clone the repo.
2. Inside the directory, create a .env file with the below mentioned fields:

TWILIO_AUTH_TOKEN=""
TWILIO_ACCOUNT_SID=""  
OPENAI_API_KEY = "" # your openai key
MY_NUMBER = "whatsapp:+81123456789" # add your number here with country code.
TWILIO_NUMBER = "whatsapp:+14987654321" # twilio number you recieve text from.

3. Create a webhook endpoint with ngrok. Setup with your credentials once signed up to their website ('https://ngrok.com/'). On System, install ngrok, and configure your account settings. Then run "ngrok http 8000". If everything went well, get the forwarding url
4. Create a twilio account. Get the SID , API_TOKEN. Save it in the .env file.
5. Then go to twillio messeging services (sandbox), activate whatsapp messaging, there enter your phone number and follow the steps which should lead to sending and recieving whatsapp message from twilio.
6. Get the twilio account number. Save it in .env file. In the senndbox settings paste the webhook url (forwarding url from step 2) also adding the endpoint like given --> "....ngrok-free.app/hook_chat" in the webhook area mentioned.
7. Open terminal inside clone repo. Create Virtual Environment. Activate it.
8. RUN in terminal : pip install -r Requirements.txt
9. RUN in terminal : uvicorn main:app

Hopefully should be able to send messages. Send "Yessir" from whatsapp to the twilio account. If you get back bullyEYE , then everything upto this step went well.

AAlthough too many steps. Need to reduce them. !! Hopefully im alive and motivated in the future to improve this. Lets see.

### Examples

Will soon update with whatsapp messages ss !

### TODO

#### Agent

1. Create query agent only for documents from a given Dir, like csv_agent.
2. Feedback mechanism.
3. Write better memory module.

#### Overall

1. Dockerize it.
2. Write better comments.
3. Write Better readme

Can reach me out at : shuvraneelroy@gmail.com
Would love to get feedback. !!
