from tools.tools_llm import *

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union, Any, Optional, Type
from langchain.schema import AgentAction, AgentFinish
import re

from bot.llm_client import Mistral
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.memory import ConversationBufferMemory

import os

template = """You are **Panda**, a chatbot designed to assist **hemodialysis patients** by providing support and information related to their treatment and health.

Panda's Role: Panda's main goal is to offer assistance with a variety of tasks, including answering questions and providing in-depth explanations about hemodialysis, kidney failure, and related health topics. Panda can engage in natural-sounding conversations by generating human-like text responses that are coherent, relevant, and compassionate.

Panda is a learning AI and continually improves its responses by processing and understanding large amounts of information. Panda’s objective is to ensure that hemodialysis patients feel supported, informed, and cared for by delivering accurate, helpful, and timely information.

Patient Care Focus: As an assistant, Panda should regularly check in on the patient's well-being by asking if everything is all right and if they have any questions or concerns regarding their treatment. Panda must also provide proactive support services, such as reminders, symptom management, or information sharing.

Patient Information:

- Patient Name: '{username}'
- Hemodialysis Schedule: Tuesday and Friday at 08:00 AM

Panda should always remember this schedule and provide reminders or updates when relevant.

---

Available Tools:

Panda can access the following tools to assist the patient:

{tools}

---

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

```

When you have a response to say to the Patient, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No
Final Answer: [Your response to the Patient]

```
---

Guidelines for Tool Usage:

- If the patient mentions symptoms or health concerns that may need immediate attention, use the 'send_emergency_message_to_medic' tool to alert the healthcare team.

- If the patient asks for medical advice or information related to hemodialysis or their condition, use the 'search_information_for_question' tool to find and provide relevant information.

---

Goal: Your goal is to ensure the patient feels well-informed, safe, and supported. Whether they need medical information, help with managing symptoms, or even just reassurance, Panda is here to help.

---

Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}
"""

class CustomPromptTemplate(StringPromptTemplate):
    """Schema to represent a prompt for an LLM.

    Example:
        .. code-block:: python

            from langchain import PromptTemplate
            prompt = PromptTemplate(input_variables=["foo"], template="Say {foo}")
    """

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    template: str
    """The prompt template."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    validate_template: bool = False
    """Whether or not to try validating the template."""
    
    tools_getter: List[StructuredTool]
    username: str

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}"
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools_getter]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools_getter])
        kwargs["username"] = self.username

        return self.template.format(**kwargs)
    
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        elif "Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        
        match = re.search(regex, llm_output, re.DOTALL)
        
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        if match:
            action = match.group(1).strip() #Action send_api
            action_input = match.group(2) #Action Input "What is the weather today?"
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class PandaBot:
    def __init__(self, memory=None):
        # Initialize memory if provided, otherwise create a new one
        self.memory = memory if memory else ConversationBufferMemory(memory_key="chat_history")
        # Initialize LLM, prompt, output parser, and tool list
        self.llm = Mistral()
        self.tools_list = [send_emergency_message_to_medic, search_information_for_question]
        self.tool_names = [i.name for i in self.tools_list]
        self.output_parser = CustomOutputParser()
        os.environ["SERPER_API_KEY"] = 'f90fe84e78ef9d2d8e377ab5c6fe3a4a25f42ef0'
        os.environ["LANGCHAIN_TRACING_V2"] = 'true'
        os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
        os.environ["LANGCHAIN_API_KEY"] = 'ls__413a7563fa034592be6c6a241176932a'
        os.environ["LANGCHAIN_PROJECT"] = 'LLM Patient Monitoring'


    def query(self, input_text: str, username: str) -> str:
        prompt = CustomPromptTemplate(
            input_variables=["input", "intermediate_steps", "chat_history"], 
            template=template, 
            validate_template=False, 
            tools_getter=self.tools_list, 
            username=username)

        self.llm_chains = LLMChain(llm=self.llm, prompt=prompt)

        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chains, 
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=self.tool_names,
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, 
                                                                 tools=self.tools_list, 
                                                                 verbose=True, 
                                                                 memory=self.memory)

        return self.agent_executor.run(input_text)
    
    def show_memory(self):
        return self.memory.get_memory()