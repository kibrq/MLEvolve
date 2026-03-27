from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodegenRequest:
    input_artifacts: dict[str, Any]
    temperature: float | None
    mode: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


class CodegenProvider(ABC):
    @abstractmethod
    def generate_artifacts(self, request: CodegenRequest, cfg) -> dict[str, Any]:
        raise NotImplementedError
