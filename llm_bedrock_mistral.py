from typing import Optional, List

import boto3
import llm
import json
from pydantic import Field, field_validator

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

# Much of this code is derived from https://github.com/sblakey/llm-bedrock-anthropic 


@llm.hookimpl
def register_models(register):
    register(
        BedrockMistral("mistral.mistral-7b-instruct-v0:2"),
        aliases=("bedrock-mistral-7b", "bmi"),
    )
    register(
        BedrockMistral("mistral.mixtral-8x7b-instruct-v0:1"),
        aliases=("bedrock-mistral-8x7b", "bm8x7bi"),
    )



class BedrockMistral(llm.Model):
    can_stream: bool = False

    # TODO: expose other Options
    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "Determines the sampling temperature. Higher values like 0.8 increase randomness, "
                "while lower values like 0.2 make the output more focused and deterministic."
            ),
            ge=0,
            le=1,
            default=0.7,
        )

        top_p: Optional[float] = Field(
            description=(
                "Nucleus sampling, where the model considers the tokens with top_p probability mass. "
                "For example, 0.1 means considering only the tokens in the top 10% probability mass."
            ),
            ge=0,
            le=1,
            default=1,
        )

        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=1024,  
        )

        @field_validator("temperature")
        def validate_temperature(cls, temperature):
            if not (0 <= temperature <= 1):
                raise ValueError("temperature must be in range 0-1")
            return temperature

        @field_validator("top_p")
        def validate_top_p(cls, top_p):
            if not (0 <= top_p <= 1):
                raise ValueError("top_p must be in range 0-1")
            return top_p

        @field_validator("max_tokens")
        def validate_length(cls, max_tokens):
            if not (0 < max_tokens <= 1_000_000):
                raise ValueError("max_tokens_to_sample must be in range 1-1,000,000")
            return max_tokens

    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
            return messages
        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        if prompt.system and self.model_id in [
            "mistral.mistral-7b-instruct-v0:2",
            "mistral.mixtral-8x7b-instruct-v0:1",
        ]:
            prompt.prompt = prompt.system + "\n" + prompt.prompt
            

        prompt.messages = self.build_messages(prompt, conversation)

        body = {
            "max_tokens": prompt.options.max_tokens,
            "prompt": f'<s>[INST]{json.dumps(prompt.messages)}[/INST]',
            "top_p": prompt.options.top_p,
            "temperature": prompt.options.temperature
        }

        encoded_data = json.dumps(body)
        prompt.prompt_json = encoded_data

        client = boto3.client('bedrock-runtime')
        if stream:
            bedrock_response = client.invoke_model_with_response_stream(
                modelId=self.model_id, body=prompt.prompt_json
            )
            chunks = bedrock_response.get("body")

            for event in chunks:
                chunk = event.get("chunk")
                response = json.loads(chunk.get("bytes").decode())
                if response["type"] == "content_block_delta":
                    completion = response["delta"]["text"]
                    yield completion

        else:
            accept = "application/json"
            contentType = "application/json"

            bedrock_response = client.invoke_model(
                modelId=self.model_id, 
                body=prompt.prompt_json,
                accept=accept,
                contentType=contentType

            )
            
            body = bedrock_response["body"].read()
            response.response_json = json.loads(body)
            completion = response.response_json["outputs"][-1]["text"]
            yield completion