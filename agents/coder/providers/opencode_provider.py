from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import difflib
import re
import time
from copy import deepcopy
from pathlib import Path

from llm import compile_prompt_to_md

from .base import CodegenProvider, CodegenRequest


logger = logging.getLogger("MLEvolve")


def _resolve_override(name: str, fallback: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    return fallback


def _default_model_id(model: str) -> str:
    if model.count("/") >= 2:
        return model.split("/", 1)[1]
    return model


PROMPT_PREFIX = (
    "Read the task and workspace context from TASK.md and the other files in the current working directory.\n\n"
    "You may do data exploration and lightweight validation work in this temporary workspace. "
    "You may inspect files, run Python smoke tests, and use very small data subsets to verify the solution behaves correctly. "
    "Use the `python_exec` MCP tool for Python execution; do not try to run Python through the bash tool. "
    "You may create multiple temporary development files while working and test separate components of the solution independently. "
    "You should check that the pipeline can correctly produce full-scale predictions using your model, even if the model is randomly initialized or minimally trained during smoke tests. "
    "At the end, compose everything into final_code.py, which must run the full-scale machine learning pipeline for this task. "
    "Do not specify explicit per-command timeouts when using the bash tool; rely on the environment's default time budgeting so the overall 12-hour MLEvolve pipeline can allocate time efficiently across exploration and optimization. "
    "Use the provided submission validator tools when relevant to verify that generated submissions are structurally correct. "
    "You have to use `bash ./validate_submission.sh` (do not forget 'bash') for submission validation. Do not finish your work until you see Validation Passed if your code should produce submission.csv."
    "Do not run full training or expensive end-to-end experiments. "
    "When you generate a solution, make it flexible: by default, running `python solution.py` should execute the full-scale solution, "
    "but the code should also support a small-scale or smoke-test mode for quick validation.\n\n"
)

PROMPT_SUFFIX = (
    "\n\nBefore returning your final answer you are obliged to check that your answer is valid. "
    "You can do smoke tests under a minute that do small runs on a very small data subset to make sure your solution works before returning. "
    "You can do data exploration and run Python smoke tests in this temporary workspace. "
    "If one of the required artifacts is code, write the code to final_code.py. "
    "If one of the required artifacts is plan, explanation, or descriptive text, write it to final_text.md. "
    "For diff tasks, write the fully updated code to final_code.py and the plan or explanation to final_text.md; MLEvolve will derive the diff automatically. "
    'Your final message should state that the artifacts are written and then report the validation checks you performed.'
)


class OpenCodeCodegenProvider(CodegenProvider):
    def generate_artifacts(self, request: CodegenRequest, cfg) -> dict[str, str]:
        if "diff" in request.mode:
            raise NotImplementedError(
                "OpenCodeCodegenProvider does not support diff modes. "
                "Disable diff mode with agent.use_diff_mode=false to use full rewrites."
            )
        if "stepwise" in request.mode:
            raise NotImplementedError(
                "OpenCodeCodegenProvider does not support stepwise modes. "
                "Disable stepwise generation with agent.use_stepwise_generation=false to use full rewrites."
            )
        prompt = request.input_artifacts.get("prompt", "")
        if not isinstance(prompt, str):
            prompt = compile_prompt_to_md(prompt)
        prompt = self._build_runtime_prompt(request.mode)
        prompt += "\n\nRead TASK.md first, then inspect the referenced artifact files before making changes.\n\n"

        if "debug" in request.mode:
            prompt += (
                "This is a bug-fixing task. First identify how to reproduce the bug in the temporary workspace. "
                "Then implement the fix. Then validate that the bug is no longer present with a focused reproduction or smoke test. "
                "Your final validation report should explicitly state how the bug was reproduced, what was changed, and what check shows the bug is gone.\n\n"
            )

        prompt += PROMPT_SUFFIX

        prompt = "Read TASK.md and implement the task staisfying all of the requirements from the SYSTEM message."

        codegen_cfg = getattr(cfg.agent, "codegen", None)
        command = list(getattr(codegen_cfg, "opencode_command", ["opencode", "exec"]))
        timeout_seconds = int(getattr(codegen_cfg, "opencode_timeout_seconds", 600))
        session_timeout_seconds = int(getattr(codegen_cfg, "max_codegen_session_timeout_seconds", 180))
        max_retry_per_session = int(getattr(codegen_cfg, "max_retry_per_opencode_session", 2))
        workspace_root = Path(cfg.workspace_dir)
        workspace_root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["MLEVOLVE_OPENCODE_BASE_URL"] = _resolve_override(
            "MLEVOLVE_OPENCODE_BASE_URL",
            cfg.agent.code.base_url,
        )
        env["MLEVOLVE_OPENCODE_API_KEY"] = _resolve_override(
            "MLEVOLVE_OPENCODE_API_KEY",
            cfg.agent.code.api_key,
        )
        env["MLEVOLVE_OPENCODE_MODEL_ID"] = _resolve_override(
            "MLEVOLVE_OPENCODE_MODEL_ID",
            _default_model_id(cfg.agent.code.model),
        )
        env["MLEVOLVE_OPENCODE_MODEL_NAME"] = _resolve_override(
            "MLEVOLVE_OPENCODE_MODEL_NAME",
            "MLEvolve Code Model",
        )
        env["OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS"] = _resolve_override(
            "OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS",
            "60000",
        )
        env["MLEVOLVE_OPENCODE_PYTHON_TIMEOUT_MS"] = _resolve_override(
            "MLEVOLVE_OPENCODE_PYTHON_TIMEOUT_MS",
            "60000",
        )
        env["MLEVOLVE_OPENCODE_PYTHON_MCP_PATH"] = _resolve_override(
            "MLEVOLVE_OPENCODE_PYTHON_MCP_PATH",
            str(Path(__file__).resolve().parents[3] / "tools" / "opencode_python_mcp.py"),
        )

        with tempfile.TemporaryDirectory(prefix="opencode-", dir=workspace_root) as temp_dir:
            temp_path = Path(temp_dir)
            self._write_workspace_inputs(temp_path, request.input_artifacts, request.mode)
            self._prepend_python_bin(env)
            data_link = temp_path / "input"
            data_source = Path(cfg.data_dir)
            if not data_link.exists():
                data_link.symlink_to(data_source, target_is_directory=data_source.is_dir())
            validator_source = Path("/home/validate_submission.sh")
            validator_link = temp_path / "validate_submission.sh"
            if validator_source.exists() and not validator_link.exists():
                validator_link.symlink_to(validator_source)
            validation_server_url = env.get("MLEVOLVE_VALIDATION_SERVER_URL")
            if validation_server_url:
                (temp_path / "VALIDATION_SERVER_URL.txt").write_text(validation_server_url + "\n")
            (temp_path / "submission").mkdir(exist_ok=True)
            (temp_path / "working").mkdir(exist_ok=True)

            self._log_preflight(temp_path, env, command)
            stdout, stderr = self._run_with_session_retries(
                temp_path=temp_path,
                env=env,
                command=command,
                initial_prompt=prompt,
                mode=request.mode,
                cfg=cfg,
                total_timeout_seconds=timeout_seconds,
                session_timeout_seconds=session_timeout_seconds,
                max_retry_per_session=max_retry_per_session,
            )

            return self._build_output_artifacts(temp_path, stdout, stderr, request.mode)

    def _run_command(
        self,
        argv: list[str],
        *,
        temp_path: Path,
        env: dict[str, str],
        timeout_seconds: int,
    ) -> tuple[str, str]:
        proc = subprocess.Popen(
            argv,
            cwd=temp_path,
            text=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def _stream(pipe, chunks: list[str], is_stderr: bool) -> None:
            assert pipe is not None
            for line in pipe:
                chunks.append(line)
                if is_stderr:
                    print(line, end="", file=sys.stderr, flush=True)
                else:
                    print(line, end="", flush=True)
            pipe.close()

        stdout_thread = threading.Thread(
            target=_stream,
            args=(proc.stdout, stdout_chunks, False),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream,
            args=(proc.stderr, stderr_chunks, True),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try:
                returncode = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncode = proc.wait()
        stdout_thread.join()
        stderr_thread.join()

        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)

        if timed_out:
            raise RuntimeError(
                f"opencode command timed out after {timeout_seconds} seconds"
            )
        if returncode != 0:
            raise RuntimeError(
                f"opencode command failed with exit code {returncode}: {stderr.strip()}"
            )
        return stdout, stderr

    def _run_with_session_retries(
        self,
        *,
        temp_path: Path,
        env: dict[str, str],
        command: list[str],
        initial_prompt: str,
        mode: str,
        cfg,
        total_timeout_seconds: int,
        session_timeout_seconds: int,
        max_retry_per_session: int,
    ) -> tuple[str, str]:
        overall_deadline = time.monotonic() + total_timeout_seconds
        stdout_total = ""
        stderr_total = ""
        last_reason = ""
        session_idx = 0

        while time.monotonic() < overall_deadline:
            session_idx += 1
            session_start = time.monotonic()
            if session_idx > 1:
                logger.warning(
                    "Starting a new opencode session after exhausting retries for previous session; session=%s",
                    session_idx,
                )
                self._reset_generated_artifacts(temp_path)

            initial_timeout = self._remaining_timeout(
                overall_deadline=overall_deadline,
                session_start=session_start,
                session_timeout_seconds=session_timeout_seconds,
            )
            stdout, stderr = self._run_command(
                [*command, initial_prompt],
                temp_path=temp_path,
                env=env,
                timeout_seconds=initial_timeout,
            )
            stdout_total += stdout
            stderr_total += stderr

            retry_count = 0
            while True:
                reason, retry_prompt, force_fresh_session = self._get_retry_reason(temp_path, env, mode, cfg)
                if not reason:
                    return stdout_total, stderr_total

                last_reason = reason
                if force_fresh_session:
                    logger.warning(
                        "opencode retry requested for mode=%s session=%s as a fresh build session: %s",
                        mode,
                        session_idx,
                        reason,
                    )
                    initial_prompt = (
                        initial_prompt
                        + "\n\nAdditional feedback from the previous attempt:\n"
                        + retry_prompt
                    )
                    break
                if retry_count >= max_retry_per_session:
                    logger.warning(
                        "opencode session retry limit reached for mode=%s after %s retries: %s",
                        mode,
                        retry_count,
                        reason,
                    )
                    break
                if time.monotonic() - session_start >= session_timeout_seconds:
                    logger.warning(
                        "opencode session timeout reached for mode=%s after %s retries: %s",
                        mode,
                        retry_count,
                        reason,
                    )
                    break

                logger.warning(
                    "opencode retry requested for mode=%s session=%s retry=%s: %s",
                    mode,
                    session_idx,
                    retry_count + 1,
                    reason,
                )
                retry_timeout = self._remaining_timeout(
                    overall_deadline=overall_deadline,
                    session_start=session_start,
                    session_timeout_seconds=session_timeout_seconds,
                )
                retry_stdout, retry_stderr = self._run_command(
                    [*command, "--continue", retry_prompt],
                    temp_path=temp_path,
                    env=env,
                    timeout_seconds=retry_timeout,
                )
                stdout_total += retry_stdout
                stderr_total += retry_stderr
                retry_count += 1

        raise RuntimeError(
            "opencode code generation exceeded total timeout "
            f"({total_timeout_seconds} seconds). Last failure reason: {last_reason or 'no additional detail'}"
        )

    def _remaining_timeout(
        self,
        *,
        overall_deadline: float,
        session_start: float,
        session_timeout_seconds: int,
    ) -> int:
        overall_remaining = overall_deadline - time.monotonic()
        session_remaining = session_timeout_seconds - (time.monotonic() - session_start)
        timeout = int(max(1, min(overall_remaining, session_remaining)))
        return timeout

    def _reset_generated_artifacts(self, temp_path: Path) -> None:
        for name in ("final_code.py", "final_text.md", "final_answer.md", "final_diff.txt"):
            path = temp_path / name
            if path.exists():
                path.unlink()
        for dirname in ("submission", "working"):
            path = temp_path / dirname
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(exist_ok=True)

    def _get_retry_reason(self, temp_path: Path, env: dict[str, str], mode: str, cfg) -> tuple[str, str, bool]:
        missing = self._missing_required_artifacts(temp_path, mode)
        if missing:
            reason = "One or more required output artifacts are missing: " + ", ".join(missing) + "."
            prompt = (
                "Your generated artifacts failed provider checks. "
                + reason
                + " Please fix exactly the failed artifact or validation issue described above, then rerun your validation."
            )
            return reason, prompt, False

        reason = self._validate_generated_workspace(temp_path, env, cfg)
        if not reason:
            return "", "", False

        if reason.startswith("Reviewer agent rejected the solution after automatic checks passed."):
            prompt = (
                "The reviewer rejected the previous solution. "
                "Start a fresh build attempt and address the reviewer feedback carefully. "
                + reason
            )
            return reason, prompt, True

        if (
            "submission/submission.csv" in reason
            or "Submission quality check failed" in reason
            or "Submission is valid" in reason
        ):
            prompt = (
                "Your generated artifacts failed provider checks. "
                + reason
                + " In other words: running `python3 final_code.py --check` produced a submission that then failed `bash ./validate_submission.sh submission/submission.csv`, "
                "or otherwise failed the submission-quality checks. Your code should produce a valid submission regardless of the `--check` flag; `--check` is allowed to be smaller and faster, "
                "but it must still use the real pipeline and produce a valid full test-set submission artifact. "
                "Do not hand-edit, patch, or synthesize `submission/submission.csv` directly. "
                "Do not generate the CSV by a fake shortcut, placeholder logic, constant predictions, cached rows, templated rows, or any synthetic path that bypasses the true preprocessing, model inference, postprocessing, or flattening logic. "
                "Instead, first remove the current submission artifact, then regenerate it by running the real pipeline through `final_code.py --check`, and then validate the regenerated submission with `bash ./validate_submission.sh submission/submission.csv`. "
                "After `--check`, the regenerated submission should already be valid."
            )
            return reason, prompt, False

        prompt = (
            "Your generated artifacts failed provider checks. "
            + reason
            + " Please fix exactly the failed artifact or validation issue described above, then rerun your validation."
        )
        return reason, prompt, False

    def _write_workspace_inputs(
        self,
        temp_path: Path,
        input_artifacts: dict[str, object],
        mode: str,
    ) -> None:
        sanitized_artifacts = self._sanitize_workspace_artifacts(input_artifacts)
        file_map = {
            "execution_output": "EXECUTION_OUTPUT.txt",
            "planning_result": "PLANNING_RESULT.md",
            "data_preview": "DATA_PREVIEW.md",
            "memory": "MEMORY.md",
            "reference_solution": "REFERENCE_SOLUTION.md",
            "branch_trajectory": "BRANCH_TRAJECTORY.md",
        }
        for key, filename in file_map.items():
            value = sanitized_artifacts.get(key)
            if value is None:
                continue
            text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=True)
            (temp_path / filename).write_text(text)
        task_sections: list[str] = []
        manifest_entries: list[str] = []
        task_description = sanitized_artifacts.get("task_description")
        if task_description is not None:
            task_sections.append("# Task Description\n")
            task_sections.append(task_description if isinstance(task_description, str) else json.dumps(task_description, indent=2, ensure_ascii=True))
        for key, filename in file_map.items():
            value = sanitized_artifacts.get(key)
            if value is not None:
                manifest_entries.append(f"- {key}: {filename}")
        for key, value in sanitized_artifacts.items():
            if value is None or key in file_map or key in {"task_description", "parent_code"}:
                continue
            filename = self._artifact_filename(key, value)
            text = self._artifact_text(key, value)
            (temp_path / filename).write_text(text)
            manifest_entries.append(f"- {key}: {filename}")
        if manifest_entries:
            task_sections.append("\n# Artifact Files\n")
            task_sections.extend(manifest_entries)
        if task_sections:
            (temp_path / "TASK.md").write_text("\n".join(task_sections).strip() + "\n")
        parent_code = sanitized_artifacts.get("parent_code")
        if parent_code is not None:
            parent_text = (
                parent_code
                if isinstance(parent_code, str)
                else json.dumps(parent_code, indent=2, ensure_ascii=True)
            )
            (temp_path / ".mlevolve_original_parent.py").write_text(parent_text)
            if "diff" in mode:
                (temp_path / "final_code.py").write_text(parent_text)
            else:
                (temp_path / "PARENT.py").write_text(parent_text)

    def _artifact_filename(self, key: str, value: object) -> str:
        stem = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_") or "artifact"
        suffix = ".md" if isinstance(value, str) else ".json"
        return f"{stem}{suffix}"

    def _artifact_text(self, key: str, value: object) -> str:
        if isinstance(value, str):
            return value
        if key == "instructions":
            return compile_prompt_to_md(value, 2)
        return json.dumps(value, indent=2, ensure_ascii=True)

    def _build_runtime_prompt(self, mode: str) -> str:
        return PROMPT_PREFIX

    def _sanitize_workspace_artifacts(self, input_artifacts: dict[str, object]) -> dict[str, object]:
        artifacts = deepcopy(input_artifacts)
        artifacts.pop("prompt", None)
        artifacts.pop("diff_instructions", None)

        instructions = artifacts.get("instructions")
        if isinstance(instructions, dict):
            cleaned: dict[str, object] = {}
            for key, value in instructions.items():
                normalized = key.strip().lower()
                if "response format" in normalized:
                    continue
                cleaned[key] = value
            artifacts["instructions"] = cleaned

        for key in ("assistant_context", "introduction"):
            value = artifacts.get(key)
            if isinstance(value, str):
                artifacts[key] = self._strip_legacy_format_language(value)

        return artifacts

    def _strip_legacy_format_language(self, text: str) -> str:
        lines: list[str] = []
        for line in text.splitlines():
            lower = line.lower()
            if "markdown code block" in lower:
                continue
            if "search/replace" in lower:
                continue
            if "response format:" in lower:
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _prepend_python_bin(self, env: dict[str, str]) -> None:
        ml_bin = "/home/nonroot/.micromamba/envs/ml/bin"
        env["PATH"] = ml_bin + os.pathsep + env.get("PATH", "")

    def _log_preflight(
        self,
        temp_path: Path,
        env: dict[str, str],
        command: list[str],
    ) -> None:
        logger.info(
            "opencode preflight env: base_url=%s model_id=%s model_name=%s bash_timeout_ms=%s path_prefix=%s command=%s",
            env.get("MLEVOLVE_OPENCODE_BASE_URL"),
            env.get("MLEVOLVE_OPENCODE_MODEL_ID"),
            env.get("MLEVOLVE_OPENCODE_MODEL_NAME"),
            env.get("OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS"),
            env.get("PATH", "").split(os.pathsep)[0] if env.get("PATH") else "",
            command,
        )
        for binary in ("python", "python3"):
            try:
                resolved = subprocess.run(
                    ["which", binary],
                    cwd=temp_path,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                path = resolved.stdout.strip() or resolved.stderr.strip() or "<not found>"
                logger.info("opencode preflight which %s -> %s", binary, path)
            except Exception as exc:
                logger.warning("opencode preflight which %s failed: %s", binary, exc)

    def _validate_generated_workspace(self, temp_path: Path, env: dict[str, str], cfg) -> str:
        from engine.validation import validate_submission_content_quality

        final_code_path = temp_path / "final_code.py"
        if not final_code_path.exists():
            return "Required artifact final_code.py is missing."

        code_text = final_code_path.read_text()
        if re.search(r"\btime\.sleep\s*\(", code_text) or re.search(r"(?<![A-Za-z0-9_])sleep\s*\(", code_text):
            return (
                "Detected an artificial sleep call in final_code.py (`time.sleep(...)` or `sleep(...)`). "
                "Do not use artificial delays to satisfy the runtime check. "
                "If your final pipeline is genuinely too fast, make the default path more substantive in legitimate ways: "
                "increase training epochs, use the full dataset, increase model capacity where appropriate, "
                "or run several training attempts with different random seeds, try a small hyperparameter search on top, "
                "and ensemble or select the best result."
            )

        check_result = self._run_final_code_check(temp_path, env, final_code_path, cfg)
        if check_result:
            return check_result

        submission_path = temp_path / "submission" / "submission.csv"
        if not submission_path.exists():
            return (
                "Expected submission/submission.csv to exist after running the --check validation path, but it does not. "
                "After `final_code.py --check`, the submission should already be generated and valid through the real pipeline."
            )

        validator_result = self._run_submission_validator(temp_path, env, submission_path)
        if validator_result:
            return validator_result

        quality_ok, quality_msg = validate_submission_content_quality(submission_path)
        if not quality_ok:
            return f"Submission quality check failed: {quality_msg}"

        reviewer_result = self._run_reviewer_verification(temp_path, env, cfg)
        if reviewer_result:
            return reviewer_result

        return ""

    def _run_final_code_check(
        self,
        temp_path: Path,
        env: dict[str, str],
        final_code_path: Path,
        cfg,
    ) -> str:
        python_cmd = list(getattr(cfg.exec, "python_cmd", ["python3"]))
        submission_dir = temp_path / "submission"
        if submission_dir.exists():
            shutil.rmtree(submission_dir)
        submission_dir.mkdir(exist_ok=True)

        try:
            result = subprocess.run(
                [*python_cmd, str(final_code_path), "--check"],
                cwd=temp_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return (
                f"Running `{ ' '.join(python_cmd) } final_code.py --check` timed out after 60 seconds. "
                "This is worrying: the `--check` path should stay lightweight enough to finish comfortably within the validation timeout "
                "while still exercising the real pipeline on a smaller scale. "
                f"stdout:\n{(exc.stdout or '').strip()}\n\nstderr:\n{(exc.stderr or '').strip()}"
            )
        if result.returncode != 0:
            return (
                f"Running `{ ' '.join(python_cmd) } final_code.py --check` failed with exit code {result.returncode}. "
                "The `--check` path should provide a lightweight but real validation run that exercises the actual pipeline and produces a valid submission artifact. "
                f"stdout:\n{result.stdout.strip()}\n\nstderr:\n{result.stderr.strip()}"
            )
        return ""

    def _run_reviewer_verification(self, temp_path: Path, env: dict[str, str], cfg) -> str:
        self._prepare_workspace_for_reviewer(temp_path)
        codegen_cfg = getattr(cfg.agent, "codegen", None)
        command = list(getattr(codegen_cfg, "opencode_command", ["opencode", "run"]))
        review_prompt = (
            "Review the generated solution artifacts in this workspace. "
            "Automatic checks already passed. "
            "Perform additional lightweight verification and put your review in the final message."
        )
        try:
            stdout, stderr = self._run_command(
                [*command, "--agent", "review", review_prompt],
                temp_path=temp_path,
                env=env,
                timeout_seconds=120,
            )
        except Exception as exc:
            return f"Reviewer agent execution failed: {exc}"

        for _ in range(2):
            report = self._extract_reviewer_message(stdout, stderr)
            if report:
                verdict = self._parse_reviewer_verdict(report)
                if verdict == "PASS":
                    return ""
                if verdict == "FAIL":
                    return (
                        "Reviewer agent rejected the solution after automatic checks passed. "
                        f"Review report:\n{report}"
                    )
                retry_prompt = (
                    "Your previous final message was malformed. "
                    "Put the review in the final message only. "
                    "The first line must be exactly `REVIEW_STATUS: PASS` or `REVIEW_STATUS: FAIL`, "
                    "followed by the concise review report."
                )
            else:
                retry_prompt = (
                    "Your previous final message was empty. "
                    "Put the review in the final message only. "
                    "The first line must be exactly `REVIEW_STATUS: PASS` or `REVIEW_STATUS: FAIL`, "
                    "followed by the concise review report."
                )
            try:
                stdout, stderr = self._run_command(
                    [*command, "--agent", "review", retry_prompt],
                    temp_path=temp_path,
                    env=env,
                    timeout_seconds=120,
                )
            except Exception as exc:
                return f"Reviewer agent execution failed during resume: {exc}"

        report = self._extract_reviewer_message(stdout, stderr)
        if not report:
            return "Reviewer agent produced an empty final message."
        verdict = self._parse_reviewer_verdict(report)
        if verdict == "PASS":
            return ""
        if verdict == "FAIL":
            return (
                "Reviewer agent rejected the solution after automatic checks passed. "
                f"Review report:\n{report}"
            )
        return (
            "Reviewer agent produced a final message, but the first line was not a valid review status. "
            f"Review report:\n{report}"
        )

    def _extract_reviewer_message(self, stdout: str, stderr: str) -> str:
        for stream in (stdout, stderr):
            text = stream.strip()
            if not text:
                continue
            idx = text.rfind("REVIEW_STATUS:")
            if idx != -1:
                return text[idx:].strip()
        return stdout.strip() or stderr.strip()

    def _parse_reviewer_verdict(self, report: str) -> str | None:
        for line in report.splitlines():
            normalized = line.strip().upper()
            if not normalized:
                continue
            if normalized == "REVIEW_STATUS: PASS":
                return "PASS"
            if normalized == "REVIEW_STATUS: FAIL":
                return "FAIL"
        upper_report = report.upper()
        if "REVIEW_STATUS: PASS" in upper_report:
            return "PASS"
        if "REVIEW_STATUS: FAIL" in upper_report:
            return "FAIL"
        if re.search(r"\bPASS\b", upper_report):
            return "PASS"
        if re.search(r"\bFAIL\b", upper_report):
            return "FAIL"
        return None

    def _prepare_workspace_for_reviewer(self, temp_path: Path) -> None:
        final_code_path = temp_path / "final_code.py"
        if final_code_path.exists():
            final_code_path.chmod(0o444)

    def _run_submission_validator(
        self,
        temp_path: Path,
        env: dict[str, str],
        submission_path: Path,
    ) -> str:
        validator_path = temp_path / "validate_submission.sh"
        if not validator_path.exists():
            return "Expected ./validate_submission.sh to exist for submission validation, but it does not."

        result = subprocess.run(
            ["bash", str(validator_path), str(submission_path)],
            cwd=temp_path,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        combined = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part).strip()
        if result.returncode != 0:
            return f"`bash ./validate_submission.sh submission/submission.csv` failed with exit code {result.returncode}: {combined}"
        if "Submission is valid" not in combined:
            return (
                "`bash ./validate_submission.sh submission/submission.csv` did not report 'Submission is valid'. "
                f"Output was: {combined}"
            )
        return ""

    def _build_output_artifacts(
        self,
        temp_path: Path,
        stdout: str,
        stderr: str,
        mode: str,
    ) -> dict[str, str]:
        final_text_path = temp_path / "final_text.md"
        final_code_path = temp_path / "final_code.py"
        final_diff_path = temp_path / "final_diff.txt"
        final_answer_path = temp_path / "final_answer.md"
        parent_path = temp_path / ".mlevolve_original_parent.py"

        plan = final_text_path.read_text().strip() if final_text_path.exists() else ""
        code = final_code_path.read_text().strip() if final_code_path.exists() else ""
        diff = final_diff_path.read_text().strip() if final_diff_path.exists() else ""
        parent_code = parent_path.read_text().strip() if parent_path.exists() else ""

        if "diff" in mode and code and parent_code:
            derived_diff = self._build_search_replace_diff(parent_code, code)
            effective_plan = plan or "MISSING PLAN"
            result = f"{effective_plan}\n\n{derived_diff}".strip() if effective_plan else derived_diff
            return {
                "result": result,
                "plan": effective_plan,
                "diff": derived_diff,
                "code": code,
                "stdout": stdout,
                "stderr": stderr,
            }

        if "diff" in mode and diff:
            effective_plan = plan or "MISSING PLAN"
            result = f"{effective_plan}\n\n{diff}".strip() if effective_plan else diff
            return {
                "result": result,
                "plan": effective_plan,
                "diff": diff,
                "stdout": stdout,
                "stderr": stderr,
            }

        if code:
            effective_plan = plan or "MISSING PLAN"
            result = (
                f"{effective_plan}\n\n```python\n{code}\n```".strip()
                if effective_plan
                else f"```python\n{code}\n```"
            )
            return {
                "result": result,
                "plan": effective_plan,
                "code": code,
                "stdout": stdout,
                "stderr": stderr,
            }

        if final_answer_path.exists():
            result = final_answer_path.read_text().strip()
            return {
                "result": result,
                "stdout": stdout,
                "stderr": stderr,
            }

        fallback = stdout.strip()
        return {
            "result": fallback,
            "stdout": stdout,
            "stderr": stderr,
        }

    def _missing_required_artifacts(self, temp_path: Path, mode: str) -> list[str]:
        missing: list[str] = []
        if "diff" in mode:
            if not (temp_path / "final_code.py").exists():
                missing.append("final_code.py")
            if not (temp_path / "final_text.md").exists():
                missing.append("final_text.md")
            return missing

        if not (temp_path / "final_code.py").exists():
            missing.append("final_code.py")
        if not (temp_path / "final_text.md").exists():
            missing.append("final_text.md")
        return missing

    def _invalid_diff_retry_note(self, temp_path: Path, parent_code: str) -> str:
        from agents.coder.diff_coder.patcher import SearchReplacePatcher

        final_code_path = temp_path / "final_code.py"
        if not final_code_path.exists():
            return (
                "The required updated code artifact final_code.py is missing. "
                "Please write the fully updated code to final_code.py and write the plan or explanation to final_text.md."
            )

        updated_code = final_code_path.read_text().strip()
        if not updated_code:
            return (
                "The required updated code artifact final_code.py is empty. "
                "Please write the fully updated code to final_code.py and write the plan or explanation to final_text.md."
            )

        if updated_code == parent_code:
            return (
                "The updated code in final_code.py is identical to the original code, so there is no effective diff to derive. "
                "Please make the intended edits in final_code.py and write the plan or explanation to final_text.md."
            )

        derived_diff = self._build_search_replace_diff(parent_code, updated_code)
        if not derived_diff.strip():
            return (
                "MLEvolve could not derive a valid diff from the original code and final_code.py. "
                "Please ensure final_code.py contains the fully updated code and write the plan or explanation to final_text.md."
            )

        patcher = SearchReplacePatcher()
        try:
            patched_code, count = patcher.apply_patch(derived_diff, parent_code, strict=False)
        except Exception as exc:
            return (
                f"MLEvolve failed to derive an applicable diff from final_code.py due to an error: {exc}. "
                "Please keep the intended updated code in final_code.py, "
                "and write the plan or explanation to final_text.md."
            )

        if count <= 0 or not patched_code or patched_code != updated_code:
            return (
                "MLEvolve derived a diff from final_code.py, but reapplying it did not reconstruct final_code.py exactly. "
                "Please simplify or stabilize the edits in final_code.py, "
                "and write the plan or explanation to final_text.md."
            )

        return ""

    def _build_search_replace_diff(self, original_text: str, updated_text: str) -> str:
        if original_text == updated_text:
            return ""

        original_lines = original_text.splitlines()
        updated_lines = updated_text.splitlines()
        matcher = difflib.SequenceMatcher(None, original_lines, updated_lines)
        blocks: list[str] = []

        for group in matcher.get_grouped_opcodes(n=1):
            original_start = max(0, group[0][1] - 1)
            original_end = min(len(original_lines), group[-1][2] + 1)
            updated_start = max(0, group[0][3] - 1)
            updated_end = min(len(updated_lines), group[-1][4] + 1)

            search = "\n".join(original_lines[original_start:original_end]).strip("\n")
            replace = "\n".join(updated_lines[updated_start:updated_end]).strip("\n")
            if search == replace:
                continue

            blocks.append(
                "\n".join(
                    [
                        "<<<<<<< SEARCH",
                        search,
                        "=======",
                        replace,
                        ">>>>>>> REPLACE",
                    ]
                )
            )

        return "\n\n".join(blocks).strip()
