"""OpenAI-compatible proxy server for chat.z.ai + Toolify-style function calling."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import secrets
import string
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpcore
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from main import ZaiClient

# ── Logging ──────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
HTTP_DEBUG = os.getenv("HTTP_DEBUG", "0") == "1"
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("zai.openai")
if not HTTP_DEBUG:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Session Pool ─────────────────────────────────────────────────────


class SessionPool:
    """Manages shared auth state; requests use per-request clients for consistency."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._authed = False
        self._token: str | None = None
        self._user_id: str | None = None
        self._username: str | None = None

    async def close(self) -> None:
        # No long-lived upstream client to close.
        return

    async def _auth_locked(self) -> None:
        """Authenticate and update snapshot. Caller must hold self._lock."""
        client = ZaiClient()
        try:
            data = await client.auth_as_guest()
            self._token = data.get("token")
            self._user_id = data.get("id")
            self._username = data.get("name") or data.get("email", "").split("@")[0]
            self._authed = bool(self._token and self._user_id)
        finally:
            await client.close()

    async def ensure_auth(self) -> None:
        # Double-check locking to avoid duplicate concurrent guest auth on cold start.
        if self._authed:
            return
        async with self._lock:
            if self._authed:
                return
            await self._auth_locked()

    async def refresh_auth(self) -> None:
        async with self._lock:
            await self._auth_locked()

    def get_auth_snapshot(self) -> dict[str, str]:
        if not (self._authed and self._token and self._user_id):
            raise RuntimeError("Auth snapshot requested before initialization")
        return {
            "token": self._token,
            "user_id": self._user_id,
            "username": self._username or "",
        }

    async def get_models(self) -> list | dict:
        await self.ensure_auth()
        auth = self.get_auth_snapshot()
        client = ZaiClient()
        try:
            client.token = auth["token"]
            client.user_id = auth["user_id"]
            client.username = auth["username"]
            return await client.get_models()
        finally:
            await client.close()


pool = SessionPool()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await pool.ensure_auth()
    yield
    await pool.close()


app = FastAPI(lifespan=lifespan)


# ── Toolify-style helpers ────────────────────────────────────────────


def _generate_trigger_signal() -> str:
    chars = string.ascii_letters + string.digits
    rand = "".join(secrets.choice(chars) for _ in range(4))
    return f"<Function_{rand}_Start/>"


GLOBAL_TRIGGER_SIGNAL = _generate_trigger_signal()


def _extract_text_from_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        return " ".join(parts).strip()
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _build_tool_call_index_from_messages(messages: list[dict]) -> dict[str, dict[str, str]]:
    idx: dict[str, dict[str, str]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
            name = str(fn.get("name", ""))
            args = fn.get("arguments", "{}")
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args = "{}"
            if isinstance(tc_id, str) and name:
                idx[tc_id] = {"name": name, "arguments": args}
    return idx


def _format_tool_result_for_ai(tool_name: str, tool_arguments: str, result_content: str) -> str:
    return (
        "<tool_execution_result>\n"
        f"<tool_name>{tool_name}</tool_name>\n"
        f"<tool_arguments>{tool_arguments}</tool_arguments>\n"
        f"<tool_output>{result_content}</tool_output>\n"
        "</tool_execution_result>"
    )


def _format_assistant_tool_calls_for_ai(tool_calls: list[dict], trigger_signal: str) -> str:
    blocks: list[str] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            args_text = args
        else:
            try:
                args_text = json.dumps(args, ensure_ascii=False)
            except Exception:
                args_text = "{}"
        blocks.append(
            "<function_call>\n"
            f"<name>{name}</name>\n"
            f"<args_json>{args_text}</args_json>\n"
            "</function_call>"
        )
    if not blocks:
        return ""
    return f"{trigger_signal}\n<function_calls>\n" + "\n".join(blocks) + "\n</function_calls>"


def _preprocess_messages(messages: list[dict]) -> list[dict]:
    tool_idx = _build_tool_call_index_from_messages(messages)
    out: list[dict] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "tool":
            tc_id = msg.get("tool_call_id")
            content = _extract_text_from_content(msg.get("content", ""))
            info = tool_idx.get(tc_id, {"name": msg.get("name", "unknown_tool"), "arguments": "{}"})
            out.append(
                {
                    "role": "user",
                    "content": _format_tool_result_for_ai(info["name"], info["arguments"], content),
                }
            )
            continue

        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
            xml_calls = _format_assistant_tool_calls_for_ai(msg["tool_calls"], GLOBAL_TRIGGER_SIGNAL)
            content = (_extract_text_from_content(msg.get("content", "")) + "\n" + xml_calls).strip()
            out.append({"role": "assistant", "content": content})
            continue

        if role == "developer":
            cloned = dict(msg)
            cloned["role"] = "system"
            out.append(cloned)
            continue

        out.append(msg)

    return out


def _generate_function_prompt(tools: list[dict], trigger_signal: str) -> str:
    tool_lines: list[str] = []
    for i, t in enumerate(tools):
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function", {}) if isinstance(t.get("function"), dict) else {}
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        desc = str(fn.get("description", "")).strip() or "None"
        params = fn.get("parameters", {})
        required = params.get("required", []) if isinstance(params, dict) else []
        try:
            params_json = json.dumps(params, ensure_ascii=False)
        except Exception:
            params_json = "{}"

        tool_lines.append(
            f"{i+1}. <tool name=\"{name}\">\n"
            f"   Description: {desc}\n"
            f"   Required: {', '.join(required) if isinstance(required, list) and required else 'None'}\n"
            f"   Parameters JSON Schema: {params_json}"
        )

    tools_block = "\n\n".join(tool_lines) if tool_lines else "(no tools)"

    return (
        "You have access to tools.\n\n"
        "When you need to call tools, you MUST output exactly:\n"
        f"{trigger_signal}\n"
        "<function_calls>\n"
        "  <function_call>\n"
        "    <name>tool_name</name>\n"
        "    <args_json>{\"arg\":\"value\"}</args_json>\n"
        "  </function_call>\n"
        "</function_calls>\n\n"
        "Rules:\n"
        "1) args_json MUST be valid JSON object\n"
        "2) For multiple calls, output one <function_calls> with multiple <function_call> children\n"
        "3) If no tool is needed, answer normally\n\n"
        f"Available tools:\n{tools_block}"
    )


def _safe_process_tool_choice(tool_choice: Any, tools: list[dict]) -> str:
    if tool_choice is None:
        return ""

    if isinstance(tool_choice, str):
        if tool_choice == "required":
            return "\nIMPORTANT: You MUST call at least one tool in your next response."
        if tool_choice == "none":
            return "\nIMPORTANT: Do not call tools. Answer directly."
        return ""

    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {}) if isinstance(tool_choice.get("function"), dict) else {}
        name = fn.get("name")
        if isinstance(name, str) and name:
            return f"\nIMPORTANT: You MUST call this tool: {name}"

    return ""


def _flatten_messages_for_zai(messages: list[dict]) -> list[dict]:
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        content = _extract_text_from_content(msg.get("content", ""))
        parts.append(f"<{role}>{content}</{role}>")
    return [{"role": "user", "content": "\n".join(parts)}]


def _remove_think_blocks(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        if start == -1:
            break
        pos = start + 7
        depth = 1
        while pos < len(text) and depth > 0:
            if text[pos : pos + 7] == "<think>":
                depth += 1
                pos += 7
            elif text[pos : pos + 8] == "</think>":
                depth -= 1
                pos += 8
            else:
                pos += 1
        if depth == 0:
            text = text[:start] + text[pos:]
        else:
            break
    return text


def _find_last_trigger_signal_outside_think(text: str, trigger_signal: str) -> int:
    if not text or not trigger_signal:
        return -1
    i = 0
    depth = 0
    last = -1
    while i < len(text):
        if text.startswith("<think>", i):
            depth += 1
            i += 7
            continue
        if text.startswith("</think>", i):
            depth = max(0, depth - 1)
            i += 8
            continue
        if depth == 0 and text.startswith(trigger_signal, i):
            last = i
            i += 1
            continue
        i += 1
    return last


def _parse_function_calls_xml(xml_string: str, trigger_signal: str) -> list[dict]:
    if not xml_string or trigger_signal not in xml_string:
        return []

    cleaned = _remove_think_blocks(xml_string)
    pos = cleaned.rfind(trigger_signal)
    if pos == -1:
        return []

    sub = cleaned[pos:]
    m = re.search(r"<function_calls>([\s\S]*?)</function_calls>", sub)
    if not m:
        return []

    calls_block = m.group(1)
    chunks = re.findall(r"<function_call>([\s\S]*?)</function_call>", calls_block)
    out: list[dict] = []

    for c in chunks:
        name_m = re.search(r"<name>([\s\S]*?)</name>", c)
        args_m = re.search(r"<args_json>([\s\S]*?)</args_json>", c)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        args_raw = (args_m.group(1).strip() if args_m else "{}")
        try:
            parsed = json.loads(args_raw) if args_raw else {}
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
        except Exception:
            parsed = {"raw": args_raw}

        out.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(parsed, ensure_ascii=False)},
            }
        )

    return out


# ── OpenAI response helpers ──────────────────────────────────────────


def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:29]}"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 2))


def _build_usage(prompt_text: str, completion_text: str) -> dict:
    p = _estimate_tokens(prompt_text)
    c = _estimate_tokens(completion_text)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _openai_chunk(
    completion_id: str,
    model: str,
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _extract_upstream_tool_calls(data: dict) -> list[dict]:
    # Native Toolify/Z.ai style
    tcs = data.get("tool_calls")
    if isinstance(tcs, list):
        return tcs

    # OpenAI-like style: choices[0].delta.tool_calls or choices[0].message.tool_calls
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0] if isinstance(choices[0], dict) else {}
        delta = c0.get("delta") if isinstance(c0.get("delta"), dict) else {}
        message = c0.get("message") if isinstance(c0.get("message"), dict) else {}
        for candidate in (delta.get("tool_calls"), message.get("tool_calls")):
            if isinstance(candidate, list):
                return candidate

    return []


def _extract_upstream_delta(data: dict) -> tuple[str, str]:
    """Best-effort extract (phase, delta_text) from upstream event payload."""
    phase = str(data.get("phase", "") or "")

    # OpenAI-like envelope
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0] if isinstance(choices[0], dict) else {}
        delta_obj = c0.get("delta") if isinstance(c0.get("delta"), dict) else {}
        msg_obj = c0.get("message") if isinstance(c0.get("message"), dict) else {}
        if not phase:
            phase = str(c0.get("phase", "") or "")
        for v in (
            delta_obj.get("reasoning_content"),
            delta_obj.get("content"),
            msg_obj.get("reasoning_content"),
            msg_obj.get("content"),
        ):
            if isinstance(v, str) and v:
                return phase, v

    candidates = [
        data.get("delta_content"),
        data.get("content"),
        data.get("delta"),
        (data.get("message") or {}).get("content") if isinstance(data.get("message"), dict) else None,
    ]

    for v in candidates:
        if isinstance(v, str) and v:
            return phase, v

    return phase, ""


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/v1/models")
async def list_models():
    models_resp = await pool.get_models()
    if isinstance(models_resp, dict) and "data" in models_resp:
        models_list = models_resp["data"]
    elif isinstance(models_resp, list):
        models_list = models_resp
    else:
        models_list = []

    return {
        "object": "list",
        "data": [
            {
                "id": m.get("id") or m.get("name", "unknown"),
                "object": "model",
                "created": 0,
                "owned_by": "z.ai",
            }
            for m in models_list
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    model: str = body.get("model", "glm-5")
    messages: list[dict] = body.get("messages", [])
    stream: bool = body.get("stream", False)
    tools: list[dict] | None = body.get("tools")
    tool_choice = body.get("tool_choice")

    # signature prompt: last user message in original request
    prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            prompt = _extract_text_from_content(msg.get("content", ""))
            break
    if not prompt:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No user message found in messages", "type": "invalid_request_error"}},
        )

    processed_messages = _preprocess_messages(messages)

    has_fc = bool(tools)
    if has_fc:
        fc_prompt = _generate_function_prompt(tools or [], GLOBAL_TRIGGER_SIGNAL)
        fc_prompt += _safe_process_tool_choice(tool_choice, tools or [])
        processed_messages.insert(0, {"role": "system", "content": fc_prompt})

    flat_messages = _flatten_messages_for_zai(processed_messages)
    usage_prompt_text = "\n".join(_extract_text_from_content(m.get("content", "")) for m in processed_messages)

    req_id = f"req_{uuid.uuid4().hex[:10]}"
    logger.info(
        "[entry][%s] model=%s stream=%s tools=%d input_messages=%d flat_chars=%d est_prompt_tokens=%d",
        req_id,
        model,
        stream,
        len(tools or []),
        len(messages),
        len(flat_messages[0].get("content", "")),
        _estimate_tokens(usage_prompt_text),
    )

    async def run_once(auth: dict[str, str]):
        client = ZaiClient()
        try:
            client.token = auth["token"]
            client.user_id = auth["user_id"]
            client.username = auth["username"]
            chat = await client.create_chat(prompt, model)
            chat_id = chat["id"]
            upstream = client.chat_completions(
                chat_id=chat_id,
                messages=flat_messages,
                prompt=prompt,
                model=model,
                tools=None,
            )
            return upstream, client
        except Exception:
            await client.close()
            raise

    if stream:

        async def gen_sse():
            completion_id = _make_id()
            retried = False

            while True:
                client: ZaiClient | None = None
                try:
                    await pool.ensure_auth()
                    auth = pool.get_auth_snapshot()
                    upstream, client = await run_once(auth)

                    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"

                    reasoning_parts: list[str] = []
                    answer_parts: list[str] = []
                    native_tool_calls: list[dict] = []

                    async for data in upstream:
                        phase, delta = _extract_upstream_delta(data)

                        upstream_tcs = _extract_upstream_tool_calls(data)
                        if upstream_tcs:
                            for tc in upstream_tcs:
                                native_tool_calls.append(
                                    {
                                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                                        "type": "function",
                                        "function": {
                                            "name": tc.get("function", {}).get("name", ""),
                                            "arguments": tc.get("function", {}).get("arguments", ""),
                                        },
                                    }
                                )
                            continue

                        if phase == "thinking" and delta:
                            reasoning_parts.append(delta)
                            chunk = _openai_chunk(completion_id, model, reasoning_content=delta)
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        elif delta:
                            answer_parts.append(delta)

                    if native_tool_calls:
                        logger.info("[stream][%s] native_tool_calls=%d", completion_id, len(native_tool_calls))
                        for i, tc in enumerate(native_tool_calls):
                            tc_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, **tc}]}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"
                        finish = _openai_chunk(completion_id, model, finish_reason="tool_calls")
                        yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    answer_text = "".join(answer_parts)
                    logger.info(
                        "[stream][%s] collected answer_len=%d reasoning_len=%d",
                        completion_id,
                        len(answer_text),
                        len("".join(reasoning_parts)),
                    )
                    parsed = _parse_function_calls_xml(answer_text, GLOBAL_TRIGGER_SIGNAL) if has_fc else []

                    if parsed:
                        logger.info("[stream][%s] parsed_tool_calls=%d", completion_id, len(parsed))
                        prefix_pos = _find_last_trigger_signal_outside_think(answer_text, GLOBAL_TRIGGER_SIGNAL)
                        if prefix_pos > 0:
                            prefix = answer_text[:prefix_pos].rstrip()
                            if prefix:
                                yield f"data: {json.dumps(_openai_chunk(completion_id, model, content=prefix), ensure_ascii=False)}\n\n"

                        for i, tc in enumerate(parsed):
                            tc_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, **tc}]}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"

                        finish = _openai_chunk(completion_id, model, finish_reason="tool_calls")
                        yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    if answer_text:
                        yield f"data: {json.dumps(_openai_chunk(completion_id, model, content=answer_text), ensure_ascii=False)}\n\n"
                    else:
                        # Never return an empty stream response body to clients.
                        yield f"data: {json.dumps(_openai_chunk(completion_id, model, content=''), ensure_ascii=False)}\n\n"

                    finish = _openai_chunk(completion_id, model, finish_reason="stop")
                    yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                except (httpcore.ReadTimeout, httpx.ReadTimeout) as e:
                    logger.error("[stream][%s] read timeout: %s", completion_id, e)
                    if client is not None:
                        await client.close()
                        client = None
                    
                    if retried:
                        # 第二次超时，返回友好错误
                        error_msg = "上游服务响应超时，请稍后重试或减少消息长度"
                        yield f"data: {json.dumps(_openai_chunk(completion_id, model, content=f'[{error_msg}]'), ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps(_openai_chunk(completion_id, model, finish_reason='error'), ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    # 第一次超时，刷新认证并重试
                    retried = True
                    logger.info("[stream][%s] retrying after timeout...", completion_id)
                    await pool.refresh_auth()
                    continue
                except Exception as e:
                    logger.exception("[stream][%s] exception: %s", completion_id, e)
                    if client is not None:
                        await client.close()
                        client = None
                    
                    if retried:
                        yield f"data: {json.dumps({'error': {'message': 'Upstream Zai error after retry', 'type': 'server_error'}}, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    retried = True
                    logger.info("[stream][%s] refreshing auth and retrying...", completion_id)
                    await pool.refresh_auth()
                    continue
                finally:
                    if client is not None:
                        await client.close()

        return StreamingResponse(
            gen_sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    completion_id = _make_id()
    client: ZaiClient | None = None
    
    for attempt in range(2):
        try:
            await pool.ensure_auth()
            auth = pool.get_auth_snapshot()
            upstream, client = await run_once(auth)
            reasoning_parts: list[str] = []
            answer_parts: list[str] = []
            native_tool_calls: list[dict] = []

            async for data in upstream:
                phase, delta = _extract_upstream_delta(data)

                upstream_tcs = _extract_upstream_tool_calls(data)
                if upstream_tcs:
                    for tc in upstream_tcs:
                        native_tool_calls.append(
                            {
                                "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                                "type": "function",
                                "function": {
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", ""),
                                },
                            }
                        )
                elif phase == "thinking" and delta:
                    reasoning_parts.append(delta)
                elif delta:
                    answer_parts.append(delta)

            if native_tool_calls:
                message: dict = {"role": "assistant", "content": None, "tool_calls": native_tool_calls}
                if reasoning_parts:
                    message["reasoning_content"] = "".join(reasoning_parts)
                usage = _build_usage(usage_prompt_text, "".join(reasoning_parts))
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                    "usage": usage,
                }

            answer_text = "".join(answer_parts)
            parsed = _parse_function_calls_xml(answer_text, GLOBAL_TRIGGER_SIGNAL) if has_fc else []
            if parsed:
                prefix_pos = _find_last_trigger_signal_outside_think(answer_text, GLOBAL_TRIGGER_SIGNAL)
                prefix_text = answer_text[:prefix_pos].rstrip() if prefix_pos > 0 else None
                message = {"role": "assistant", "content": prefix_text or None, "tool_calls": parsed}
                if reasoning_parts:
                    message["reasoning_content"] = "".join(reasoning_parts)
                usage = _build_usage(usage_prompt_text, (prefix_text or "") + "".join(reasoning_parts))
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                    "usage": usage,
                }

            usage = _build_usage(usage_prompt_text, answer_text + "".join(reasoning_parts))
            msg: dict = {"role": "assistant", "content": answer_text}
            if reasoning_parts:
                msg["reasoning_content"] = "".join(reasoning_parts)
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                "usage": usage,
            }

        except Exception as e:
            logger.exception("[sync][%s] exception: %s", completion_id, e)
            if client is not None:
                await client.close()
                client = None
            
            if attempt == 0:
                await pool.refresh_auth()
                continue
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "Upstream Zai error after retry", "type": "server_error"}},
            )
        finally:
            if client is not None:
                await client.close()

    return JSONResponse(status_code=502, content={"error": {"message": "Unexpected error", "type": "server_error"}})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30016)
