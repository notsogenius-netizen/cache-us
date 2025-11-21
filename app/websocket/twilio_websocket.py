"""
Twilio WebSocket handler for Media Streams.
"""
import json
import logging
import asyncio
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import websockets
from websockets.exceptions import ConnectionClosedError

from app.utils.context_manager import context_manager
from app.services.deepgram_service import deepgram_service
from app.services.llm_service import llm_service
from app.services.rag_service import rag_service
from app.services.tts_service import tts_service
from app.services.tool_executor import tool_executor

logger = logging.getLogger(__name__)


class TwilioWebSocketHandler:
    """Handler for Twilio Media Streams WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def handle_connection(self, websocket: WebSocket, call_sid: Optional[str] = None):
        """
        Handle WebSocket connection from Twilio.

        Args:
            websocket: FastAPI WebSocket connection
            call_sid: Call SID from Twilio (optional, can be extracted from messages)
        """
        await websocket.accept()
        logger.info(f"[Twilio] WebSocket connected, call_sid: {call_sid}")
        # Extract phone_number and Stream parameters from Twilio messages
        extracted_phone_number = None
        scheduled_call_id = None
        user_name = None
        pending_messages = []  # Store messages we read while looking for phone_number
        
        # Try to get phone_number and Stream parameters from initial Twilio messages
        try:
            # Wait for initial "connected" or "start" event from Twilio
            initial_message = await websocket.receive()
            pending_messages.append(initial_message)
            
            if "text" in initial_message:
                try:
                    data = json.loads(initial_message["text"])
                    event_type = data.get("event", "")
                    
                    if event_type in ["connected", "start"]:
                        # Extract phone_number from Twilio Stream message
                        extracted_phone_number = (
                            data.get("phoneNumber") 
                            or data.get("phone_number")
                            or data.get("callSid")  # Sometimes phone number is in callSid
                        )
                        
                        # Extract Stream parameters (from <Parameter> tags in TwiML)
                        stream_params = data.get("streamParams", {})
                        if isinstance(stream_params, dict):
                            scheduled_call_id = stream_params.get("scheduled_call_id")
                            user_name = stream_params.get("user_name")
                        else:
                            # Try direct access
                            scheduled_call_id = data.get("scheduled_call_id")
                            user_name = data.get("user_name")
                            
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Error reading initial message: {e}")

        # If still no phone_number, try to get from "start" event
        if not extracted_phone_number:
            try:
                # Wait a bit for start event
                start_message = await asyncio.wait_for(websocket.receive(), timeout=2.0)
                pending_messages.append(start_message)
                
                if "text" in start_message:
                    try:
                        data = json.loads(start_message["text"])
                        if data.get("event") == "start":
                            extracted_phone_number = (
                                data.get("phoneNumber") 
                                or data.get("phone_number")
                                or data.get("callSid")
                            )
                            # Extract Stream parameters
                            stream_params = data.get("streamParams", {})
                            if isinstance(stream_params, dict):
                                scheduled_call_id = stream_params.get("scheduled_call_id") or scheduled_call_id
                                user_name = stream_params.get("user_name") or user_name
                    except:
                        pass
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

        if not extracted_phone_number:
            # Phone number is required to find agent
            await websocket.close(code=1008, reason="Phone number not found in Twilio Stream")
            return

        deepgram_connection = None
        try:
            # Get or create conversation context
            context = await context_manager.get_or_create_context(
                phone_number=extracted_phone_number, websocket=websocket
            )

            if not context:
                # Agent not found for this phone number
                await websocket.close(code=1008, reason="Agent not found")
                return

            # Store active connection with call_sid or phone_number as key
            connection_key = call_sid or extracted_phone_number
            self.active_connections[connection_key] = websocket

            # Initialize Deepgram connection for STT
            transcribed_text_buffer = []
            current_transcription = ""

            async def on_transcript(text: str):
                """Handle Deepgram transcription."""
                nonlocal current_transcription
                current_transcription = text
                transcribed_text_buffer.append(text)

            async def on_transcript_error(error: Exception):
                """Handle Deepgram error."""
                print(f"Deepgram error: {error}")

            # Create Deepgram connection
            deepgram_connection = await deepgram_service.create_live_transcription(
                on_transcript=on_transcript,
                on_error=on_transcript_error,
            )

            # Process messages from Twilio (including any pending messages we read earlier)
            message_queue = pending_messages.copy()
            
            while True:
                try:
                    # Get message from queue or receive new one
                    if message_queue:
                        message = message_queue.pop(0)
                    else:
                        message = await websocket.receive()

                    # Handle different message types
                    if "bytes" in message:
                        # Binary audio data - send to Deepgram
                        audio_bytes = message["bytes"]
                        await deepgram_service.send_audio(
                            deepgram_connection, audio_bytes
                        )

                    elif "text" in message:
                        # JSON metadata from Twilio
                        try:
                            data = json.loads(message["text"])
                            event_type = data.get("event", "")

                            if event_type == "connected":
                                # Connection established
                                print(f"Twilio connected: {extracted_phone_number}")
                                # Extract Stream parameters if available
                                stream_params = data.get("streamParams", {})
                                if isinstance(stream_params, dict):
                                    if not scheduled_call_id:
                                        scheduled_call_id = stream_params.get("scheduled_call_id")
                                    if not user_name:
                                        user_name = stream_params.get("user_name")

                            elif event_type == "start":
                                # Call started - phone_number should be in this event
                                print(f"Call started: {extracted_phone_number}")
                                if not extracted_phone_number:
                                    extracted_phone_number = data.get("phoneNumber") or data.get("phone_number")
                                # Extract Stream parameters if available
                                stream_params = data.get("streamParams", {})
                                if isinstance(stream_params, dict):
                                    if not scheduled_call_id:
                                        scheduled_call_id = stream_params.get("scheduled_call_id")
                                    if not user_name:
                                        user_name = stream_params.get("user_name")

                            elif event_type == "media":
                                # Media stream data (if not using binary)
                                # Handle if needed
                                pass

                            elif event_type == "stop":
                                # Call ended
                                print(f"Call stopped: {extracted_phone_number}")
                                break

                        except json.JSONDecodeError:
                            # Invalid JSON, ignore
                            pass

                    # Check if we have new transcribed text
                    if transcribed_text_buffer:
                        # Process the latest transcription
                        transcribed_text = transcribed_text_buffer.pop(0)
                        
                        if transcribed_text and transcribed_text.strip():
                            # Process this utterance
                            await self._process_utterance(
                                transcribed_text=transcribed_text,
                                context=context,
                            )

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"Error processing message: {e}")
                    break

        except ConnectionClosedError:
            pass
        except Exception as e:
            print(f"Error in WebSocket connection: {e}")
        finally:
            # Cleanup
            connection_key = call_sid or extracted_phone_number
            if connection_key and connection_key in self.active_connections:
                del self.active_connections[connection_key]

            # Close Deepgram connection
            if deepgram_connection:
                try:
                    deepgram_service.close_connection(deepgram_connection)
                except:
                    pass

            # Mark context as inactive (but keep for reconnection)
            if extracted_phone_number:
                context_manager.close_websocket(extracted_phone_number)

    async def _process_utterance(
        self, transcribed_text: str, context
    ):
        """
        Process a user utterance: RAG → LLM → Tool execution → TTS → Send audio.

        Args:
            transcribed_text: User's transcribed text
            context: ConversationContext
        """
        try:
            # Add user message to conversation history
            context.add_user_message(transcribed_text)

            # Query FAISS for RAG context (fresh query, no caching)
            rag_context = await rag_service.get_rag_context(
                transcribed_text=transcribed_text,
                namespace=context.namespace,
                top_k=3,
            )

            # Generate LLM response with function calling
            llm_response = await llm_service.generate_response(
                agent_prompt=context.prompt,
                transcribed_text=transcribed_text,
                rag_context=rag_context,
                conversation_history=context.conversation_history,
                tools=context.tools,
            )

            # Handle LLM response
            final_response_text = await self._handle_llm_response(
                llm_response=llm_response,
                context=context,
            )

            if final_response_text:
                # Convert to speech and send
                await self._send_tts_response(final_response_text, context)

        except Exception as e:
            print(f"Error processing utterance: {e}")
            # Send error message via TTS
            error_message = "I apologize, but I encountered an error processing your request."
            await self._send_tts_response(error_message, context)

    async def _handle_llm_response(
        self, llm_response: Dict[str, Any], context
    ) -> Optional[str]:
        """
        Handle LLM response (text or tool call).

        Args:
            llm_response: LLM response dict
            context: ConversationContext

        Returns:
            Final text response to send to user
        """
        response_type = llm_response.get("type")

        if response_type == "text":
            # Regular text response
            text = llm_response.get("content", "")
            if text:
                context.add_assistant_message(text)
            return text

        elif response_type == "tool_call":
            # LLM wants to call a tool
            tool_name = llm_response.get("tool_name")
            tool_arguments = llm_response.get("tool_arguments", {})

            # Find tool definition
            tool_definition = None
            for tool in context.tools:
                if tool.tool_name == tool_name:
                    tool_definition = tool
                    break

            if not tool_definition:
                return f"I apologize, but I couldn't find the tool '{tool_name}'."

            # Check if it's end_call tool
            if tool_name == "end_call":
                # Close WebSocket connection
                await self._end_call(context.phone_number)
                return "Goodbye!"

            # Execute tool
            tool_result = await tool_executor.execute_tool(
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                tool_definition=tool_definition,
            )

            # Format tool result as string
            if tool_result.get("status") == "error":
                tool_result_str = f"Error: {tool_result.get('error', 'Unknown error')}"
            else:
                # Format successful result
                tool_result_str = json.dumps(tool_result, indent=2)

            # Send tool result back to LLM
            llm_final_response = await llm_service.generate_response_with_tool_result(
                agent_prompt=context.prompt,
                conversation_history=context.conversation_history,
                tool_name=tool_name,
                tool_result=tool_result_str,
                tools=context.tools,
            )

            # Handle final response (could be text or another tool call)
            return await self._handle_llm_response(llm_final_response, context)

        return None

    async def _send_tts_response(self, text: str, context):
        """
        Convert text to speech and send audio to client.

        Args:
            text: Text to convert to speech
            context: ConversationContext
        """
        try:
            # Convert text to speech
            audio_bytes = await tts_service.text_to_speech(text)

            if audio_bytes:
                # Send audio bytes to Twilio via WebSocket
                websocket = context.websocket
                if websocket:
                    # Twilio expects audio in specific format
                    # For Media Streams, we may need to send in specific chunks
                    # For now, send as binary
                    await websocket.send_bytes(audio_bytes)

        except Exception as e:
            print(f"Error sending TTS response: {e}")

    async def _end_call(self, phone_number: str):
        """End the call by closing WebSocket connection."""
        if phone_number in self.active_connections:
            websocket = self.active_connections[phone_number]
            await websocket.close(code=1000, reason="Call ended by tool")


# Global instance
twilio_websocket_handler = TwilioWebSocketHandler()

