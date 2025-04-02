
import instructor
from openai import AsyncOpenAI
import os
from dataclasses import dataclass
from pydantic import BaseModel



class Ragoutput(BaseModel):
    response:str



PROMPT = "Your are expert document chatbot. Answer from following documents."


@dataclass
class AgentAI:
    model = "gpt-4o-mini"
    prompt: str = PROMPT
    

    def _read_markdown_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content

    def __post_init__(self):
        files = [os.path.join("ragas-airline-dataset",file) for file in os.listdir("ragas-airline-dataset") if file.endswith(".md")]
        self.content = [f"{self._read_markdown_file(file)}---\n\n--{file}" for file in files]
        self.content = '\n\n'.join(self.content)
        self.client = instructor.from_openai(AsyncOpenAI())


    async def ask(self, query:str):
        response = await self.client.chat.completions.create(
                        model=self.model,
                        response_model=Ragoutput,
                        messages=[
                            {"role": "system", "content": self.prompt},
                            {"role": "user", "content": self.content},
                            {"role": "user", "content": query}
                        ],
                    )

        return response.response
        

agent = AgentAI()