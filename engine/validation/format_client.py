"""Validation server client: health check and submission validate."""

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger("MLEvolve")


def _normalize_validate_response(response_json: dict) -> dict:
    """Accept both MLEvolve and original mle-bench grading server payloads."""
    if "is_valid" in response_json and "result" in response_json:
        return response_json

    if "result" in response_json:
        result = response_json["result"]
        is_valid = isinstance(result, str) and result.strip() == "Submission is valid."
        return {"is_valid": is_valid, "result": result}

    return response_json


def _normalize_validate_text(response_text: str) -> dict:
    """Accept plain-text validator responses."""
    result = response_text.strip()
    is_valid = result == "Submission is valid."
    return {"is_valid": is_valid, "result": result}


def _extract_validate_response(response: requests.Response) -> dict:
    """Accept JSON or plain-text validation responses."""
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        return _normalize_validate_response(response.json())

    text = response.text.strip()
    if not text:
        return {"is_valid": False, "result": ""}

    try:
        maybe_json: Any = response.json()
    except ValueError:
        return _normalize_validate_text(text)

    if isinstance(maybe_json, dict):
        return _normalize_validate_response(maybe_json)

    if isinstance(maybe_json, str):
        return _normalize_validate_text(maybe_json)

    return {"is_valid": False, "result": text}


def get_server_url_list():
    """Return validator base URLs.

    Defaults to the standard MLE-bench validation server.
    Legacy support remains available through GRADING_SERVER_PORT.
    """
    explicit_url = os.getenv("MLEVOLVE_VALIDATION_SERVER_URL")
    if explicit_url:
        return [explicit_url.rstrip("/")]

    server_port = os.getenv("GRADING_SERVER_PORT", "5005")
    use_legacy_server = os.getenv("MLEVOLVE_USE_LOCAL_VALIDATION_SERVER", "0") == "1"
    if use_legacy_server:
        return [f"http://127.0.0.1:{server_port}"]

    return ["http://127.0.0.1:5000"]


server_url_list = get_server_url_list()


def is_server_online(max_retries=3, timeout=300):
    server_url_list = get_server_url_list()
    retry = 0
    index = 0
    server_url = server_url_list[index]
    while retry < max_retries:
        try:
            response = requests.get(f"{server_url}/health", timeout=timeout)
            if response.status_code == 200:
                logger.info(f"Server {server_url} is online, status code: {response.status_code}")
                return True, server_url
            else:
                logger.warning(f"Server returned non-200 status code: {response.status_code}")
                logger.warning(f"Response body: {response.text[:500]}")
                logger.warning(f"Response headers: {dict(response.headers)}")
                if response.status_code in {404, 405}:
                    logger.info(
                        f"Server {server_url} does not expose /health, assuming validation endpoint may still be available."
                    )
                    return True, server_url

        except requests.exceptions.Timeout:
            timeout += 20
            logger.error(f"Connection to {server_url} timed out.")
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {server_url}, connection error.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Connection to {server_url} failed.")
        retry += 1
        if retry < max_retries:
            index += 1
            index = index%(len(server_url_list))
            server_url = server_url_list[index]
            logger.info(f"Retrying... ({retry}/{max_retries})")
            time.sleep(1)
    logger.error(f"Server is not online after {max_retries} retries.")
    return False, server_url


def call_validate(exp_id, submission_path, timeout=300, max_retries=3):
    online, server_url = is_server_online()
    retry=0
    while retry < max_retries:
        try:
            if online:
                with open(submission_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(
                        f"{server_url}/validate",
                        files=files,
                        headers={"exp-id": exp_id},
                        timeout=timeout,
                    )
                response_json = _extract_validate_response(response)
                if "error" in response_json:
                    logger.error(f"Server returned error: {response.text}")
                    return False, response_json['details']
                else:
                    return True, _normalize_validate_response(response_json)
            else:
                return False, f"Server at {server_url} is not online"
        except requests.exceptions.Timeout:
            logger.error(f"Connection to {server_url} timed out.")
            timeout += 20
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {server_url}, connection error.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Connection to {server_url} failed.")
        retry += 1
        if retry < max_retries:
            logger.info(f"Retrying... ({retry}/{max_retries})")
            time.sleep(1)
        else:
            return False, ""
