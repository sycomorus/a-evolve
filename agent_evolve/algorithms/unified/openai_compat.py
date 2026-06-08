"""OpenAI-compatible provider used by the artifact-local unified engine."""

from __future__ import annotations

import json
import os
from typing import Any

from agent_evolve.llm.base import LLMMessage, LLMResponse


class OpenAICompatProvider:
    """Small OpenAI chat-completions wrapper with tool-loop support."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        omit_temperature: bool = False,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("Install openai to use an OpenAI-compatible evolver") from exc

        self.model = model
        resolved_base_url = (
            base_url
            or os.environ.get("EVOLVER_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        resolved_api_key = (
            api_key
            or os.environ.get("EVOLVER_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        kwargs: dict[str, Any] = {}
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
            kwargs["api_key"] = resolved_api_key or "EMPTY"
        elif resolved_api_key:
            kwargs["api_key"] = resolved_api_key
        self.client = openai.OpenAI(**kwargs)
        self.temperature = temperature
        self.omit_temperature = omit_temperature

    def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **_: Any,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
        }
        if not self.omit_temperature:
            resolved_temperature = self.temperature if self.temperature is not None else temperature
            if resolved_temperature is not None:
                params["temperature"] = resolved_temperature
        response = self.client.chat.completions.create(**params)
        content, _tool_calls = self._message_parts(response)
        usage = self._response_usage(response)
        return LLMResponse(
            content=content,
            usage={
                "input_tokens": self._usage_value(usage, "prompt_tokens"),
                "output_tokens": self._usage_value(usage, "completion_tokens"),
            },
            raw=response,
        )

    def complete_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
        **_: Any,
    ) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            tools=self._to_openai_tools(tools),
        )
        content, _tool_calls = self._message_parts(response)
        usage = self._response_usage(response)
        return LLMResponse(
            content=content,
            usage={
                "input_tokens": self._usage_value(usage, "prompt_tokens"),
                "output_tokens": self._usage_value(usage, "completion_tokens"),
            },
            raw=response,
        )

    def converse_loop(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]],
        tool_executor: dict[str, Any],
        max_tokens: int = 16384,
        max_turns: int = 50,
    ) -> LLMResponse:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        openai_tools = self._to_openai_tools(tools)
        input_tokens = 0
        output_tokens = 0
        text_parts: list[str] = []
        last_response: Any = None

        for _ in range(max_turns):
            params: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if openai_tools:
                params["tools"] = openai_tools
                params["tool_choice"] = "auto"
            response = self.client.chat.completions.create(**params)
            last_response = response
            usage = self._response_usage(response)
            input_tokens += self._usage_value(usage, "prompt_tokens")
            output_tokens += self._usage_value(usage, "completion_tokens")

            content, tool_calls = self._message_parts(response)
            if content:
                text_parts.append(content)

            assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)

            if not tool_calls:
                break

            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                name = function.get("name") or ""
                raw_args = function.get("arguments") or "{}"
                try:
                    parsed_args = json.loads(raw_args)
                    executor = tool_executor.get(name)
                    if executor is None:
                        result_text = f"ERROR: Unknown tool '{name}'"
                    elif isinstance(parsed_args, dict):
                        result_text = str(executor(**parsed_args))
                    else:
                        result_text = str(executor(parsed_args))
                except Exception as exc:  # noqa: BLE001
                    result_text = f"ERROR: {exc}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": result_text,
                })

        return LLMResponse(
            content="\n".join(text_parts),
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            raw=last_response,
        )

    @staticmethod
    def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function":
                result.append(tool)
                continue
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            })
        return result

    @staticmethod
    def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
        if isinstance(tool_call, dict):
            return tool_call
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments or "{}",
            },
        }

    @classmethod
    def _message_parts(cls, response: Any) -> tuple[str, list[dict[str, Any]]]:
        raw = cls._response_payload(response)
        if isinstance(raw, str):
            return raw, []

        choices = raw.get("choices") or []
        if not choices:
            return str(raw), []

        choice = choices[0]
        if not isinstance(choice, dict):
            choice = cls._response_payload(choice)
        if not isinstance(choice, dict):
            return str(choice), []

        message = choice.get("message") or choice.get("delta") or {}
        if not isinstance(message, dict):
            message = cls._response_payload(message)
        if not isinstance(message, dict):
            return str(message), []

        content = message.get("content")
        if content is None:
            content = choice.get("text") or ""
        tool_calls = [
            cls._tool_call_to_dict(tool_call)
            for tool_call in (message.get("tool_calls") or [])
        ]
        return str(content or ""), tool_calls

    @classmethod
    def _response_usage(cls, response: Any) -> Any:
        raw = cls._response_payload(response)
        return raw.get("usage") if isinstance(raw, dict) else None

    @staticmethod
    def _response_payload(response: Any) -> Any:
        if isinstance(response, str):
            text = response.strip()
            if not text:
                return ""
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return response
        if hasattr(response, "model_dump"):
            return response.model_dump(exclude_none=True)
        return response

    @staticmethod
    def _usage_value(usage: Any, key: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return int(usage.get(key, 0) or 0)
        return int(getattr(usage, key, 0) or 0)
