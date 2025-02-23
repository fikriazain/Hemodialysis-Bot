from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from groq import Groq
import os
import requests

# HOST = '127.0.0.1:5000'
URI = f'https://api.groq.com/openai/v1/chat/completions'

client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get('GROQ_API_KEY'),
)


class Mistral(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if isinstance(stop, list):
            stop = stop + ["\n###","\nObservation:",'\nObservation:']
        
        response = client.chat.completions.create(stop=stop,temperature=0.0,max_completion_tokens=256, messages=[{
            'role': 'user',
            'content': prompt
        }], model='llama3-70b-8192', )
    
        return response.choices[0].message.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
