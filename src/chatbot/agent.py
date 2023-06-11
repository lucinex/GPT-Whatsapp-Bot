from src.conf import AGENT_DATA_PATH

# from src.chatbot.agent_tools import DocumentSearchTool
# from langchain.memory import  CombinedMemory
from src.chatbot.agent_tools import load_all_tools
from src.chatbot.modules.conversation_history import FAISSChatMemory
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from .modules.langchain_callbacks import BaseCallbackHandler
from .modules.chroma_memory import get_chroma_memory
from src.conf import AGENT_SETTINGS

# class Interface:
#     def __init__(self, path):
#         self.data_path = AGENT_DATA_PATH+"/"+path
#         self.all_tools = self.load_tools(self.data_path)
#     def load_tools(self, path):
#         doc_search_tool = DocumentSearchTool(path).generate_tool()
#         return


class Agent_BOT:
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

    def __init__(self, memory_client):
        # init agent from start ,
        # self.inititialize_all(session_name)
        self.agent_memory = AGENT_DATA_PATH
        # self.agent = self.init_agent()

        self.callback = BaseCallbackHandler()
        self.feedback_flag = False
        self.memory_client = memory_client
        self.memory = self.get_memory()

    def init_agent(self, tools):
        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=ChatOpenAI(
                temperature=0,
                model_name="gpt-4",
                max_tokens=1000,
            ),
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
            return_intermediate_steps=True,
        )
        return agent_executor

    def get_tools(self):
        more_tools = load_all_tools()
        return more_tools

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
        tools = self.get_tools()
        tool_names = [tool.name for tool in tools]
        agent_executor = self.init_agent(tools)
        with get_openai_callback() as cb:
            response = agent_executor({"input": query})  # , callbacks=[self.callback])
        im_steps = self.get_intermediate_steps_str(response)
        resp = (
            str(response["output"])
            + f"\n\n<========>\n*Analytics*: \n[Tokens: {cb.total_tokens}]\n[{im_steps}]"
        )
        if self.feedback_flag:
            self.feedback(query, str(response["output"]), im_steps)
        return resp

    def feedback(self, query: str, response: str, in_steps: str):
        memory = 1
        pass
