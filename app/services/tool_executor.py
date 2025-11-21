"""
Tool executor service for executing tools (parsing curl commands, HTTP requests, etc.).
"""
from typing import Dict, Any, Optional
import re
import shlex
import httpx
import json

from app.models.tool import Tool


class ToolExecutor:
    """Service for executing tools."""

    async def execute_tool(
        self, tool_name: str, tool_arguments: Dict[str, Any], tool_definition: Tool
    ) -> Dict[str, Any]:
        """
        Execute a tool based on its definition and arguments.

        Args:
            tool_name: Name of the tool to execute
            tool_arguments: Arguments provided by LLM
            tool_definition: Tool definition from database

        Returns:
            Dict with execution result
        """
        # Handle special tools
        if tool_name == "end_call":
            return {"action": "end_call", "status": "success"}

        # Extract curl command from tool.parameters
        curl_command = self._extract_curl_command(tool_definition.parameters)
        
        if not curl_command:
            return {
                "status": "error",
                "error": "No curl command found in tool parameters",
            }

        # Parse curl command
        parsed_request = self._parse_curl_command(curl_command)
        
        if not parsed_request:
            return {
                "status": "error",
                "error": "Failed to parse curl command",
            }

        # Substitute arguments into curl command if needed
        # For now, we'll execute the curl as-is
        # You can enhance this to substitute values from tool_arguments
        
        # Execute HTTP request
        try:
            result = await self._execute_http_request(parsed_request)
            return result
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    def _extract_curl_command(self, parameters: Any) -> Optional[str]:
        """
        Extract curl command from tool parameters.

        Args:
            parameters: Tool parameters (could be dict, string, etc.)

        Returns:
            Curl command string or None
        """
        if parameters is None:
            return None

        if isinstance(parameters, str):
            # If it's a string, check if it's a curl command
            if parameters.strip().startswith("curl"):
                return parameters.strip()
            return None

        if isinstance(parameters, dict):
            # If it's a dict, look for 'curl' key
            if "curl" in parameters:
                return str(parameters["curl"])
            # Check for curl_command, command, etc.
            for key in ["curl_command", "command", "url", "action"]:
                if key in parameters:
                    value = parameters[key]
                    if isinstance(value, str) and value.strip().startswith("curl"):
                        return value.strip()
            return None

        return None

    def _parse_curl_command(self, curl_command: str) -> Optional[Dict[str, Any]]:
        """
        Parse curl command to extract HTTP method, URL, headers, and body.

        Args:
            curl_command: Curl command string

        Returns:
            Dict with 'method', 'url', 'headers', 'data' or None if parsing fails
        """
        try:
            # Remove 'curl' prefix and trim
            command = curl_command.strip()
            if command.startswith("curl"):
                command = command[4:].strip()

            # Initialize parsed request
            parsed = {
                "method": "GET",
                "url": None,
                "headers": {},
                "data": None,
            }

            # Use regex to find URL (first quoted or unquoted string)
            url_match = re.search(r"['\"](https?://[^'\"]+)['\"]", command)
            if not url_match:
                url_match = re.search(r"(https?://\S+)", command)
            
            if url_match:
                parsed["url"] = url_match.group(1).strip('"\'')
            else:
                return None

            # Find HTTP method (-X flag)
            method_match = re.search(r"-X\s+(\w+)", command, re.IGNORECASE)
            if method_match:
                parsed["method"] = method_match.group(1).upper()

            # Find headers (-H flags)
            header_matches = re.findall(r"-H\s+['\"]([^'\"]+)['\"]", command)
            for header in header_matches:
                if ":" in header:
                    key, value = header.split(":", 1)
                    parsed["headers"][key.strip()] = value.strip()

            # Find data/body (-d or --data flag)
            data_match = re.search(r"(-d|--data)\s+['\"]([^'\"]+)['\"]", command)
            if not data_match:
                # Try without quotes
                data_match = re.search(r"(-d|--data)\s+([^\s]+)", command)
            
            if data_match:
                parsed["data"] = data_match.group(2).strip('"\'')
                # Try to parse as JSON
                try:
                    parsed["data"] = json.loads(parsed["data"])
                except:
                    pass  # Keep as string

            return parsed

        except Exception as e:
            print(f"Error parsing curl command: {e}")
            return None

    async def _execute_http_request(self, parsed_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute HTTP request using httpx.

        Args:
            parsed_request: Parsed request dict with method, url, headers, data

        Returns:
            Dict with status_code, body, headers
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=parsed_request["method"],
                    url=parsed_request["url"],
                    headers=parsed_request.get("headers", {}),
                    json=parsed_request.get("data") if isinstance(parsed_request.get("data"), dict) else None,
                    data=parsed_request.get("data") if not isinstance(parsed_request.get("data"), dict) else None,
                    timeout=30.0,
                )

                return {
                    "status": "success",
                    "status_code": response.status_code,
                    "body": response.text,
                    "headers": dict(response.headers),
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


# Global instance
tool_executor = ToolExecutor()

