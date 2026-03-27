from __future__ import annotations

from functools import lru_cache

from .base import CodegenProvider, CodegenRequest
from .llm_provider import LlmCodegenProvider
from .opencode_provider import OpenCodeCodegenProvider


@lru_cache(maxsize=None)
def _provider_for_name(name: str) -> CodegenProvider:
    normalized = (name or "llm").strip().lower()
    if normalized == "llm":
        return LlmCodegenProvider()
    if normalized == "opencode":
        return OpenCodeCodegenProvider()
    raise ValueError(f"Unsupported codegen provider '{name}'")


def get_codegen_provider(cfg) -> CodegenProvider:
    codegen_cfg = getattr(cfg.agent, "codegen", None)
    provider_name = getattr(codegen_cfg, "provider", "llm")
    return _provider_for_name(provider_name)


def generate_code(
    agent_instance,
    prompt: str | dict | list,
    *,
    mode: str = "default",
    input_artifacts: dict | None = None,
    metadata: dict | None = None,
) -> str:
    output_artifacts = generate_code_artifacts(
        agent_instance,
        input_artifacts=input_artifacts or {"prompt": prompt},
        mode=mode,
        metadata=metadata,
    )
    result = output_artifacts.get("result")
    if not isinstance(result, str):
        raise ValueError("Codegen provider must return output_artifacts['result'] as a string")
    return result


def generate_code_artifacts(
    agent_instance,
    *,
    input_artifacts: dict[str, object],
    mode: str = "default",
    metadata: dict | None = None,
) -> dict[str, object]:
    provider = get_codegen_provider(agent_instance.cfg)
    request = CodegenRequest(
        input_artifacts=input_artifacts,
        temperature=agent_instance.acfg.code.temp,
        mode=mode,
        metadata=metadata or {},
    )
    return provider.generate_artifacts(request, agent_instance.cfg)
