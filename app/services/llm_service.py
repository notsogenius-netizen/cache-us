"""
LLM service for Cerebras with function calling support.
"""
from typing import List, Dict, Optional, Any
import json
import asyncio
from cerebras.cloud.sdk import Cerebras

from app.core.config import settings
from app.models.tool import Tool


class LLMService:
    """Service for LLM interactions with function calling."""

    def __init__(self):
        self.client = Cerebras(
            api_key=settings.cerebras_api_key,
        )
        self.model = settings.llm_model

    def _format_tools_for_cerebras(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """
        Format tools for Cerebras function calling format.

        Args:
            tools: List of Tool objects

        Returns:
            List of tool definitions in Cerebras/OpenAI format
        """
        formatted_tools = []
        for tool in tools:
            # Parse tool.parameters - could be JSON schema or curl command
            # For function calling, we need JSON schema
            # If tool.parameters is a curl command, we'll still pass it
            # but the LLM needs to understand it via description
            function_def = {
                "name": tool.tool_name,
                "description": tool.description or f"Tool: {tool.tool_name}",
            }
            
            # Handle parameters - if it's a JSON schema, use it
            # If it's a curl command stored as string, create a schema
            if tool.parameters:
                if isinstance(tool.parameters, dict):
                    # If it's already a schema, use it
                    if "type" in tool.parameters or "properties" in tool.parameters:
                        function_def["parameters"] = tool.parameters
                    else:
                        # If it's a dict with curl, extract what we can
                        function_def["parameters"] = {
                            "type": "object",
                            "properties": tool.parameters,
                        }
                else:
                    # If it's a string (curl command), create minimal schema
                    function_def["parameters"] = {
                        "type": "object",
                        "properties": {
                            "curl_command": {
                                "type": "string",
                                "description": f"Cur command for {tool.tool_name}. Original: {tool.parameters}",
                            }
                        },
                    }
            else:
                function_def["parameters"] = {"type": "object", "properties": {}}

            # Format as OpenAI tools format
            formatted_tools.append({
                "type": "function",
                "function": function_def
            })

        return formatted_tools

    def _format_conversation_history(
        self, conversation_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format conversation history for OpenAI API.
        Preserves all message fields including tool_calls, function_call, etc.

        Args:
            conversation_history: List of message dicts with 'role', 'content', and optionally 'tool_calls' or 'function_call'

        Returns:
            Formatted messages for OpenAI API
        """
        formatted_messages = []
        for msg in conversation_history:
            formatted_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            # Preserve tool_calls if present (for assistant messages with tool calls)
            if "tool_calls" in msg:
                formatted_msg["tool_calls"] = msg["tool_calls"]
            # Preserve function_call if present (legacy format)
            if "function_call" in msg:
                formatted_msg["function_call"] = msg["function_call"]
            formatted_messages.append(formatted_msg)
        return formatted_messages

    async def generate_response(
        self,
        agent_prompt: str,
        transcribed_text: str,
        rag_context: List[str],
        conversation_history: List[Dict[str, str]],
        tools: List[Tool],
    ) -> Dict[str, Any]:
        """
        Generate LLM response with RAG context and function calling.

        Args:
            agent_prompt: Base agent prompt
            transcribed_text: Latest user transcribed text
            rag_context: List of RAG context chunks (top-3)
            conversation_history: Full conversation history
            tools: List of available tools

        Returns:
            Dict with 'type' ('text' or 'tool_call'), 'content' or 'tool_name'/'tool_arguments'
        """
        # Build system message with agent prompt and RAG context
        rag_context_text = "\n\n".join(rag_context) if rag_context else "No relevant context found."
        
        system_message = f"""{agent_prompt}

Context from Knowledge Base:
{rag_context_text}
"""

        # Format conversation history
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history (excluding the latest user message - we'll add it separately)
        history_messages = self._format_conversation_history(conversation_history)
        messages.extend(history_messages)
        
        # Add latest user message
        messages.append({"role": "user", "content": transcribed_text})

        # Format tools for function calling
        tools_list = self._format_tools_for_cerebras(tools)

        try:
            # Call Cerebras API (synchronous, wrap in async)
            # Cerebras SDK might use 'tools' instead of 'functions'
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
            }
            
            # Add tools if available (Cerebras may use 'tools' parameter)
            if tools_list:
                # Try 'tools' first (newer OpenAI-compatible format)
                kwargs["tools"] = tools_list
                kwargs["tool_choice"] = "auto"
            
            # Run synchronous call in thread pool
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **kwargs
            )

            choice = response.choices[0]
            message = choice.message

            # Check if LLM wants to call a function/tool
            # Cerebras/OpenAI may use tool_calls (list) or function_call (single)
            if hasattr(message, "tool_calls") and message.tool_calls:
                # New format: tool_calls is a list
                tool_call = message.tool_calls[0]
                tool_arguments = tool_call.function.arguments
                if isinstance(tool_arguments, str):
                    try:
                        tool_arguments = json.loads(tool_arguments)
                    except json.JSONDecodeError:
                        tool_arguments = {}
                
                # Extract tool_call_id if available
                tool_call_id = getattr(tool_call, "id", "call_1")
                
                return {
                    "type": "tool_call",
                    "tool_name": tool_call.function.name,
                    "tool_arguments": tool_arguments,
                    "tool_call_id": tool_call_id,
                    "content": message.content or "",
                    "assistant_message": {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": json.dumps(tool_arguments) if isinstance(tool_arguments, dict) else str(tool_arguments)
                                }
                            }
                        ]
                    }
                }
            elif hasattr(message, "function_call") and message.function_call:
                # Old format: function_call (single)
                tool_arguments = message.function_call.arguments
                if isinstance(tool_arguments, str):
                    try:
                        tool_arguments = json.loads(tool_arguments)
                    except json.JSONDecodeError:
                        tool_arguments = {}
                
                return {
                    "type": "tool_call",
                    "tool_name": message.function_call.name,
                    "tool_arguments": tool_arguments,
                    "content": message.content or "",
                }

            # Regular text response
            return {
                "type": "text",
                "content": message.content or "",
            }

        except Exception as e:
            # Log error
            print(f"Error in LLM service: {e}")
            return {
                "type": "text",
                "content": "I apologize, but I encountered an error processing your request.",
            }

    async def generate_response_with_tool_result(
        self,
        agent_prompt: str,
        conversation_history: List[Dict[str, str]],
        tool_name: str,
        tool_result: str,
        tools: List[Tool],
        tool_call_id: Optional[str] = "call_1",
    ) -> Dict[str, Any]:
        """
        Generate LLM response after tool execution.

        Args:
            agent_prompt: Base agent prompt
            conversation_history: Full conversation history
            tool_name: Name of tool that was executed
            tool_result: Result from tool execution
            tools: List of available tools

        Returns:
            Dict with 'type' and 'content'
        """
        # Build messages with tool result
        messages = [{"role": "system", "content": agent_prompt}]
        history_messages = self._format_conversation_history(conversation_history)
        messages.extend(history_messages)
        
        # Add tool execution result
        # Cerebras API expects 'tool' role, not 'function' role
        # The tool result must come after the assistant's message with the tool call
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": tool_result,
        })

        # Format tools for function calling (in case LLM wants to call another tool)
        tools_list = self._format_tools_for_cerebras(tools)

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
            }
            
            if tools_list:
                kwargs["tools"] = tools_list
                kwargs["tool_choice"] = "auto"
            
            # Run synchronous call in thread pool
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **kwargs
            )

            choice = response.choices[0]
            message = choice.message

            # Check if LLM wants to call another function/tool
            if hasattr(message, "tool_calls") and message.tool_calls:
                # New format: tool_calls is a list
                tool_call = message.tool_calls[0]
                tool_arguments = tool_call.function.arguments
                if isinstance(tool_arguments, str):
                    try:
                        tool_arguments = json.loads(tool_arguments)
                    except json.JSONDecodeError:
                        tool_arguments = {}
                
                # Extract tool_call_id if available
                tool_call_id = getattr(tool_call, "id", "call_1")
                
                return {
                    "type": "tool_call",
                    "tool_name": tool_call.function.name,
                    "tool_arguments": tool_arguments,
                    "tool_call_id": tool_call_id,
                    "content": message.content or "",
                }
            elif hasattr(message, "function_call") and message.function_call:
                # Old format: function_call (single)
                tool_arguments = message.function_call.arguments
                if isinstance(tool_arguments, str):
                    try:
                        tool_arguments = json.loads(tool_arguments)
                    except json.JSONDecodeError:
                        tool_arguments = {}
                
                return {
                    "type": "tool_call",
                    "tool_name": message.function_call.name,
                    "tool_arguments": tool_arguments,
                    "content": message.content or "",
                }

            return {
                "type": "text",
                "content": message.content or "",
            }

        except Exception as e:
            print(f"Error in LLM service (with tool result): {e}")
            return {
                "type": "text",
                "content": "I apologize, but I encountered an error processing the tool result.",
            }


# Global instance
llm_service = LLMService()

