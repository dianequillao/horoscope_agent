import os

from anthropic import Anthropic
from arize.otel import register
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.instrumentation import using_attributes
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues


from dotenv import load_dotenv

load_dotenv() 

## Code from https://arize.com/docs/ax/cookbooks/instrument/anthropic-sdk-guide#agents-built-with-the-anthropic-sdk
tracer_provider = register(
    space_id=os.environ["ARIZE_SPACE_ID"],
    api_key=os.environ["ARIZE_API_KEY"],
    project_name=os.environ.get("ARIZE_PROJECT_NAME", "anthropic-sdk-agent"),
)

tracer = tracer_provider.get_tracer(__name__)


# Auto-instrument the Anthropic SDK: each messages.create call becomes an LLM span.
AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

# Turn off the Anthropic SDK's own spans (anthropic 1.14.0 and later), so
# each request isn't traced twice. Earlier SDK versions ignore this.
os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def check_sign(birthdate):
    birthday_response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=4096,
        system='You are an astrologer, return the sign only when they ask you for their sign.',
        messages=[
            {"role": "user", "content": "Given this birth day {}, return the horoscope sign".format(birthdate)}
        ],
    )

    return next(b.text for b in birthday_response.content if b.type == "text").strip()

def get_horoscope(sign):
    horoscope_response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        messages=[
            {"role": "user", "content": "Given this sign {} return the horoscope".format(sign)}
        ],
    )
    return next(b.text for b in horoscope_response.content if b.type == "text").strip()



if __name__ == "__main__":
    # new span without a parent
    with tracer.start_as_current_span(
        "horoscope_agent",
        attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value},
    ) as span:
        span.set_attribute(SpanAttributes.INPUT_VALUE, "07/15/1990")
        sign = check_sign("07/15/1990")
        horoscope = get_horoscope(sign)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, horoscope)
        print(horoscope)