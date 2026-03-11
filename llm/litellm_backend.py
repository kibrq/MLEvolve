"""LiteLLM backend: function calling (query), streaming generation (generate),
   prompt compilation, and function-calling specs."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import jsonschema
from dataclasses_json import DataClassJsonMixin
from funcy import notnone, select_values
from litellm import completion
from config import Config

logger = logging.getLogger("MLEvolve")

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict
    description: str

    def __post_init__(self):
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
            "strict": True,
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }


def _build_messages(system_message: str | None, user_message: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    if not messages:
        raise ValueError("Either system_message or user_message must be provided")
    return messages


def _extract_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    in_tokens = getattr(usage, "prompt_tokens", 0) or 0
    out_tokens = getattr(usage, "completion_tokens", 0) or 0
    return int(in_tokens), int(out_tokens)


def _client_kwargs(base_url: str, api_key: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"timeout": 1200}
    if base_url:
        kwargs["api_base"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _parse_structured_response(response: Any, func_spec: FunctionSpec) -> dict:
    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []

    if tool_calls:
        arguments = tool_calls[0].function.arguments
        payload = json.loads(arguments) if isinstance(arguments, str) else arguments
    else:
        text = _normalize_content(getattr(message, "content", None))
        if not text:
            raise ValueError(f"No structured output returned for function {func_spec.name}")
        payload = json.loads(text)

    if isinstance(payload, list):
        if not payload:
            raise ValueError("Structured output was an empty list")
        payload = payload[0]
    return payload


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    request: dict[str, Any] = {
        "model": filtered_kwargs.get("model", "gemini/gemini-3-pro-preview"),
        "messages": _build_messages(system_message, user_message),
        "temperature": filtered_kwargs.get("temperature", 1.0),
        "max_tokens": filtered_kwargs.get("max_tokens", 16384),
        **_client_kwargs(cfg.agent.feedback.base_url, cfg.agent.feedback.api_key),
    }

    if func_spec is not None:
        request["tools"] = [func_spec.as_openai_tool_dict]
        request["tool_choice"] = func_spec.openai_tool_choice_dict
        request["response_format"] = {"type": "json_object"}

    t0 = time.time()
    logger.info(f"Querying LiteLLM with model: {request['model']}")
    response = completion(**request)
    req_time = time.time() - t0

    if func_spec is None:
        output = _normalize_content(response.choices[0].message.content)
        logger.info(f"LiteLLM response: {output}", extra={"verbose": True})
    else:
        output = _parse_structured_response(response, func_spec)
        logger.info(f"LiteLLM structured output response: {output}", extra={"verbose": True})

    in_tokens, out_tokens = _extract_usage(response)
    info = {
        "model": request["model"],
        "created": int(time.time()),
    }
    return output, req_time, in_tokens, out_tokens, info


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int = 20,
    retry_delay: float = 3,
) -> str:
    if prompt is not None and not isinstance(prompt, str):
        prompt = compile_prompt_to_md(prompt)

    logger.info(f"generate prompt: {prompt}", extra={"verbose": True})

    for attempt in range(max_retries):
        try:
            request: dict[str, Any] = {
                "model": cfg.agent.code.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature if temperature is not None else 1.0,
                "max_tokens": max_tokens if max_tokens is not None else 16384,
                "stream": True,
                **_client_kwargs(cfg.agent.code.base_url, cfg.agent.code.api_key),
            }
            if stop_tokens:
                request["stop"] = stop_tokens
            if json_schema is not None:
                request["response_format"] = {"type": "json_object"}
                logger.info("Requesting JSON output", extra={"verbose": True})

            response = completion(**request)
            full_text = ""
            for chunk in response:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                full_text += _normalize_content(getattr(delta, "content", None))

            if "</think>" in full_text:
                full_text = full_text[full_text.find("</think>") + 8:]

            logger.info(f"generate response: {full_text}", extra={"verbose": True})
            return full_text

        except Exception as e:
            logger.warning(f"generate failed, retrying {attempt + 1}/{max_retries}: {e}")
            if attempt >= max_retries - 1:
                logger.error("generate retry limit reached")
                raise
            time.sleep(retry_delay)
