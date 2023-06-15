from src.conf import AGENT_DATA_PATH

# from src.chatbot.agent_tools import DocumentSearchTool
# from langchain.memory import  CombinedMemory
from src.chatbot.agent_tools import load_all_tools

# from src.chatbot.modules.conversation_history import FAISSChatMemory
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from .modules.langchain_callbacks import BaseCallbackHandler
from .modules.chroma_memory import get_chroma_memory
from src.conf import AGENT_SETTINGS, CSV_DATA_DIR
from src.chatbot.modules.csv_tools.functionality import create_csv_agent
import os

# class Interface:
#     def __init__(self, path):
#         self.data_path = AGENT_DATA_PATH+"/"+path
#         self.all_tools = self.load_tools(self.data_path)
#     def load_tools(self, path):
#         doc_search_tool = DocumentSearchTool(path).generate_tool()
#         return
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSV_BOT:
    csv_dir = CSV_DATA_DIR

    def __init__(self):
        logger.info(f"Init CSV_BOT, {self.get_csv_files()}")
        self.llm = None
        self.csv_files = {i: j for i, j in self.get_csv_files()}
        self.csv_kwargs = {"filenames": []}

    def init_csv_agent(self, **kwargs):
        # self.reset()
        if self.csv_kwargs.get("filenames", []) == []:
            raise ValueError("No CSV paths were mentioned while initializing CSV Agent")
        else:
            filepaths = []
            fnames = self.csv_kwargs.get("filenames", [])
            logger.info(f"Loading files in agent {self.csv_kwargs}{self.csv_files}")
            for name in fnames:
                if self.csv_files.get(name, False):
                    filepaths.append(self.csv_files[name])
            if len(filepaths) > 0:
                if len(filepaths) == 1:
                    return create_csv_agent(self.llm, filepaths[0], **kwargs)
                else:
                    return create_csv_agent(self.llm, filepaths, **kwargs)

            else:
                raise FileExistsError("No CSV files to create agents from!! ")

    def set_csv_kwargs(self, filenames=[]):
        self.csv_kwargs = {"filenames": filenames}

    def reset_csv_agent(self):
        self.csv_kwargs = {"filenames": []}
        self.csv_files = {i: j for i, j in self.get_csv_files()}

    def get_csv_files(self):
        # logger.info("Getting files in CSV DIR")
        new_files = []
        if os.path.isdir(self.csv_dir):
            files = os.listdir(self.csv_dir)
            # logger.info(f"CSV Files: {files,len(files)}")
            for fil in files:
                # logger.info(f"Endswith csv {fil.endswith('.csv')}")
                if fil.endswith(".csv"):
                    # logger.info(f"Getting files in CSV DIR filesss {fil}")
                    new_files.append((fil, f"{self.csv_dir}/{fil}"))

                else:
                    continue

        return new_files


class Agent_BOT(CSV_BOT):
    prefix = f"""
You are Luci-BT. You are a large language model trained by OpenAI. 
You are able to assist with a wide range of tasks, from answering 
simple questions to planning and taking multiple actions to accomplish a task
Luci-BT is able to remember a certain period of conversations. Luci-BT uses
previous conversations with the User and picks up context from it when required. 
Luci-BT is able to break down complex tasks based on tools available to it and plan ahead before naively using any tools. 
Hence Luci-BT can produce highly complex thoughts to plan ahead. 
You are Luci-BT.
"""
    suffix = """

TOOLS
------
Luci-BT can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:
{{tools}}
{format_instructions}

NOTES:
------
When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response.

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else).
{{{{input}}}}"""

    allowed_agent_types = ["general", "csv"]
    delimiter = "#*#"

    def __init__(self, memory_client):
        super().__init__()
        # init agent from start ,
        # self.inititialize_all(session_name)
        self.agent_memory = AGENT_DATA_PATH
        # self.agent = self.init_agent()

        self.callback = BaseCallbackHandler()
        self.feedback_flag = False
        self.memory_client = memory_client
        self.memory = self.get_memory()
        self.agent_type = "general"
        self.agent_type_kwargs = {}
        self.cache = None
        self.analytics = True
        self.intermediate_steps = True
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            max_tokens=1000,
        )

        logger.info("Initialized Agent > Staring up!!")

    def init_general_agent(self):
        tools = self.get_general_tools()
        tool_names = [tool.name for tool in tools]
        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=self.llm,
            system_message=self.prefix,
            human_message=self.suffix,
            tools=tools,
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=self.memory,
            max_iterations=4,
            return_intermediate_steps=self.intermediate_steps,
        )
        return agent_executor

    def init_agent(self):
        if self.agent_type == self.allowed_agent_types[0]:
            return self.init_general_agent()
        elif self.agent_type == self.allowed_agent_types[1]:
            return self.init_csv_agent(
                return_intermediate_steps=self.intermediate_steps, verbose=True
            )

    def get_general_tools(self):
        if self.agent_type == self.allowed_agent_types[0]:
            more_tools = load_all_tools()
            more_tools += self.get_csv_agent_tool()
        elif self.agent_type == self.allowed_agent_types[1]:
            more_tools = []

        return more_tools

    def get_csv_agent_tool(self):
        name = "Chat with CSV"
        files = self.get_csv_files()
        files, filepaths = [i for i, _ in files], [i for _, i in files]
        description = (
            f"""Initialize an Intelligent Chatbot with CSVs mentioned : [{' , '.join(files)}]. Input should be a string of csv filenames (from the mentioned documents) and delimited by {self.delimiter}. Example input: "example1.csv{self.delimiter}example2.csv"""
            ""
        )

        def set_agent(filenames):
            try:
                self.set_agent_type("csv")
                files = [f.strip() for f in filenames.split(self.delimiter)]
                self.reset_csv_agent()
                self.set_csv_kwargs(files)
                logger.info(f"Set csv kwargs {self.csv_kwargs}")
                return "CSV Agent successfully initialized"
            except Exception as e:
                return f"Error while Initalizing CSV Agent, got exception: {e}"

        return [
            Tool(
                name=name,
                func=set_agent,
                description=description,
            )
        ]

    def get_memory(self):
        # memory = FAISSChatMemory().init(self.agent_memory)

        memory = get_chroma_memory(client=self.memory_client, k=3, read_only=False)
        return memory

    def get_intermediate_steps_str(self, response):
        return_str = "\Intermediate_steps were:"
        if response["intermediate_steps"]:
            for e, i in enumerate(response["intermediate_steps"]):
                action, obs = i
                return_str += f"\n\tStep {e+1}:"
                return_str += f"\nThought:{action.log}\n"
        return return_str

    def run(self, query):
        agent_executor = self.init_agent()
        with get_openai_callback() as cb:
            response = agent_executor({"input": query})  # , callbacks=[self.callback])
        im_steps = self.get_intermediate_steps_str(response)
        if self.analytics:
            resp = (
                str(response["output"])
                + f"\n\n<========>\n*Analytics*: \n[Tokens: {cb.total_tokens}]\n[{im_steps}]"
            )
        else:
            resp = str(response["output"])
        if self.feedback_flag:
            self.feedback(query, str(response["output"]), im_steps)
        return resp

    def set_agent_type(self, type: str = ""):
        if type == "":
            self.reset_agent_type(self.allowed_agent_types[0])
        elif type in self.allowed_agent_types:
            self.agent_type = type

    def reset_agent_type(self):
        self.set_agent_type()
        self.self.reset_csv_agent()
