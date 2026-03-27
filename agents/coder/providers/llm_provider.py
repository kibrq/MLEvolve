from __future__ import annotations

from llm import generate as llm_generate

from .base import CodegenProvider, CodegenRequest


class LlmCodegenProvider(CodegenProvider):
    def generate_artifacts(self, request: CodegenRequest, cfg) -> dict[str, Any]:
        prompt = request.input_artifacts.get("prompt")
        if prompt is None:
            raise ValueError("LlmCodegenProvider requires input_artifacts['prompt']")
        result = llm_generate(
            prompt=prompt,
            temperature=request.temperature,
            cfg=cfg,
        )
        return {"result": result}
