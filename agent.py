import os

from anthropic import Anthropic
from arize.otel import register
from openinference.instrumentation.anthropic import AnthropicInstrumentor

from dotenv import load_dotenv

load_dotenv() 

## Code from https://arize.com/docs/ax/cookbooks/instrument/anthropic-sdk-guide#agents-built-with-the-anthropic-sdk
tracer_provider = register(
    space_id=os.environ["ARIZE_SPACE_ID"],
    api_key=os.environ["ARIZE_API_KEY"],
    project_name=os.environ.get("ARIZE_PROJECT_NAME", "anthropic-sdk-agent"),
)

# Auto-instrument the Anthropic SDK: each messages.create call becomes an LLM span.
AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

# Turn off the Anthropic SDK's own spans (anthropic 1.14.0 and later), so
# each request isn't traced twice. Earlier SDK versions ignore this.
os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

message = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What's my horoscope today? I'm a Pisces"}
    ],
)

for block in message.content:
    if block.type == "text":
        print(block.text)