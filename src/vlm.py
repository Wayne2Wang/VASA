"""Vision-capable Chat Completions endpoint with bounded transport retries."""
from .engine.client_llm import _encode_message_images


class EmptyVLMResponse(RuntimeError):
    """A completed API response without usable text (counts toward the budget)."""


def _field(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _response_text(message):
    content = _field(message, 'content')
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text = ''.join(_field(part, 'text', '') or '' for part in content)
        if text.strip():
            return text
    for key in ('reasoning', 'reasoning_content'):
        text = _field(message, key)
        if isinstance(text, str) and text.strip():
            return text
    chunks = []
    for part in _field(message, 'reasoning_details', []) or []:
        text = _field(part, 'text') or _field(part, 'content')
        if isinstance(text, str):
            chunks.append(text)
    return '\n'.join(chunks).strip()


class VLMClient:
    def __init__(self, *, base_url, model, api_key, downscale=2, reasoning_effort=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=2, timeout=120)
        self.model = model
        self.downscale = downscale
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else ('low' if 'glm' in model.lower() else None)
        self.calls = 0
        self.usage = []

    def __call__(self, messages, max_tokens=10000):
        kwargs = dict(model=self.model, messages=_encode_message_images(messages, downscale_factor=self.downscale), max_tokens=max_tokens)
        if self.reasoning_effort:
            kwargs['extra_body'] = {'reasoning': {'effort': self.reasoning_effort}}
        response = self.client.chat.completions.create(**kwargs)
        self.calls += 1
        if response.usage:
            self.usage.append(response.usage.model_dump())
        choice = response.choices[0] if response.choices else None
        message = _field(choice, 'message')
        text = _response_text(message)
        if text:
            return text
        usage = response.usage
        details = _field(usage, 'completion_tokens_details')
        raise EmptyVLMResponse(
            f"VLM returned no usable text (model={self.model}, "
            f"finish_reason={_field(choice, 'finish_reason')!r}, "
            f"prompt_tokens={_field(usage, 'prompt_tokens')}, "
            f"completion_tokens={_field(usage, 'completion_tokens')}, "
            f"reasoning_tokens={_field(details, 'reasoning_tokens')}, "
            f"refusal={bool(_field(message, 'refusal'))}). "
            "If finish_reason is 'length', increase --max-tokens or reduce reasoning effort."
        )
