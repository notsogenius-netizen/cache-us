"""
Twilio WebSocket handler for Media Streams.
"""
import json
import logging
import asyncio
import base64
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

    async def handle_connection(self, websocket: WebSocket, call_sid: Optional[str] = None, phone_number: Optional[str] = None):
        """
        Handle WebSocket connection from Twilio.

        Args:
            websocket: FastAPI WebSocket connection
            call_sid: Call SID from Twilio (optional, can be extracted from messages)
            phone_number: Phone number from webhook (optional, will extract from stream if not provided)
        """
        await websocket.accept()
        logger.info(f"[Twilio] WebSocket connected, call_sid: {call_sid}, phone_number: {phone_number}")
        print(f"[Twilio] WebSocket connected, call_sid: {call_sid}, phone_number: {phone_number} (type: {type(phone_number)})")
        # Use phone_number from query parameter if provided, otherwise extract from Twilio messages
        # FastAPI should automatically decode URL-encoded values (e.g., %2B -> +)
        # Note: Twilio doesn't pass query params to WebSocket, so we'll get these from Stream Parameters
        # Initialize variables before use
        extracted_phone_number = phone_number
        extracted_call_sid = call_sid
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
                    
                    if event_type == "connected":
                        # Log the connected event (streamParams are usually empty here)
                        logger.info(f"[Twilio] Received connected event: {json.dumps(data, indent=2)}")
                        print(f"[Twilio] Received connected event:")
                        print(json.dumps(data, indent=2))
                        # streamParams are usually empty in "connected" event, wait for "start" event
                        
                    elif event_type == "start":
                        # Log the full message for debugging
                        logger.info(f"[Twilio] Received start event: {json.dumps(data, indent=2)}")
                        print(f"[Twilio] Received start event:")
                        print(json.dumps(data, indent=2))
                        
                        # Extract phone_number from Twilio Stream message only if not already provided
                        if not extracted_phone_number:
                            extracted_phone_number = (
                                data.get("phoneNumber") 
                                or data.get("phone_number")
                                or data.get("callSid")  # Sometimes phone number is in callSid
                            )
                        
                        # Extract Stream parameters (from <Parameter> tags in TwiML)
                        # The "start" event should have streamParams with our parameters
                        stream_params = data.get("streamParams", {})
                        logger.info(f"[Twilio] Stream params: {stream_params}")
                        print(f"[Twilio] Stream params: {stream_params}")
                        
                        if isinstance(stream_params, dict) and stream_params:
                            # Extract call_sid and phone_number from Stream parameters
                            if not extracted_call_sid:
                                extracted_call_sid = stream_params.get("call_sid")
                            if not extracted_phone_number:
                                extracted_phone_number = stream_params.get("phone_number")
                            scheduled_call_id = stream_params.get("scheduled_call_id")
                            user_name = stream_params.get("user_name")
                        else:
                            # Try direct access
                            if not extracted_call_sid:
                                extracted_call_sid = data.get("call_sid")
                            if not extracted_phone_number:
                                extracted_phone_number = data.get("phone_number")
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
                            # Only extract if not already provided from query parameter
                            if not extracted_phone_number:
                                extracted_phone_number = (
                                    data.get("phoneNumber") 
                                    or data.get("phone_number")
                                    or data.get("callSid")
                                )
                            # Extract Stream parameters
                            stream_params = data.get("streamParams", {})
                            if isinstance(stream_params, dict):
                                if not extracted_call_sid:
                                    extracted_call_sid = stream_params.get("call_sid") or extracted_call_sid
                                if not extracted_phone_number:
                                    extracted_phone_number = stream_params.get("phone_number") or extracted_phone_number
                                scheduled_call_id = stream_params.get("scheduled_call_id") or scheduled_call_id
                                user_name = stream_params.get("user_name") or user_name
                    except:
                        pass
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

        # Phone number is optional - we'll use the first available agent if not provided
        if extracted_phone_number:
            logger.info(f"[Twilio] Using phone_number: {extracted_phone_number} for agent lookup")
        else:
            logger.info("[Twilio] No phone number provided, will use first available agent from database")

        deepgram_connection = None
        stream_sid = None  # Store streamSid for sending audio back
        try:
            # Get or create conversation context
            # Pass None if phone_number not found - will use first available agent
            context = await context_manager.get_or_create_context(
                phone_number=extracted_phone_number if extracted_phone_number else None, 
                websocket=websocket
            )

            if not context:
                # Agent not found for this phone number
                await websocket.close(code=1008, reason="Agent not found")
                return

            # Store active connection with call_sid or phone_number as key
            # Use a default key if both are None
            connection_key = extracted_call_sid or extracted_phone_number or "default"
            self.active_connections[connection_key] = websocket

            # Initialize Deepgram connection for STT
            transcribed_text_buffer = []
            current_transcription = ""

            async def on_transcript(text: str):
                """Handle Deepgram transcription."""
                nonlocal current_transcription
                current_transcription = text
                transcribed_text_buffer.append(text)
                logger.info(f"[Deepgram] 📝 Transcription received: {text}")
                print(f"[Deepgram] 📝 Transcription: {text}")

            async def on_transcript_error(error: Exception):
                """Handle Deepgram error."""
                logger.error(f"[Deepgram] ❌ Error: {error}")
                print(f"[Deepgram] ❌ Error: {error}")

            # Create Deepgram connection
            deepgram_connection = await deepgram_service.create_live_transcription(
                on_transcript=on_transcript,
                on_error=on_transcript_error,
            )

            # Process messages from Twilio (including any pending messages we read earlier)
            message_queue = pending_messages.copy()
            
            # Track message counts for debugging
            binary_message_count = 0
            text_message_count = 0
            media_event_count = 0
            
            # Add periodic stats logging during the call
            import time
            last_stats_log = time.time()
            stats_log_interval = 10  # Log stats every 10 seconds
            
            while True:
                try:
                    # Get message from queue or receive new one
                    if message_queue:
                        message = message_queue.pop(0)
                    else:
                        message = await websocket.receive()
                    
                    # Handle different message types
                    if "bytes" in message:
                        binary_message_count += 1
                        # Binary audio data - send to Deepgram
                        audio_bytes = message["bytes"]
                        audio_size = len(audio_bytes)
                        logger.info(f"[Twilio] 🔊 Received binary audio data: {audio_size} bytes")
                        print(f"[Twilio] 🔊 Received binary audio data: {audio_size} bytes")
                        
                        # Log first few bytes for debugging (first 10 bytes as hex)
                        if audio_size > 0:
                            hex_preview = " ".join(f"{b:02x}" for b in audio_bytes[:10])
                            logger.debug(f"[Twilio] Audio data preview (first 10 bytes): {hex_preview}")
                            print(f"[Twilio] Audio data preview (first 10 bytes): {hex_preview}")
                        
                        # Send to Deepgram if connection exists
                        if deepgram_connection:
                            await deepgram_service.send_audio(
                                deepgram_connection, audio_bytes
                            )
                        else:
                            logger.warning("[Twilio] Deepgram connection not initialized, skipping audio")
                            print("[Twilio] ⚠️ Deepgram connection not initialized, skipping audio")

                    elif "text" in message:
                        text_message_count += 1
                        # JSON metadata from Twilio
                        try:
                            data = json.loads(message["text"])
                            event_type = data.get("event", "")
                            
                            if event_type == "media":
                                media_event_count += 1

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
                                logger.info(f"[Twilio] Start event full data: {json.dumps(data, indent=2)}")
                                print(f"[Twilio] Start event full data:")
                                print(json.dumps(data, indent=2))
                                
                                # Extract streamSid - needed for sending audio back
                                # streamSid is in the "start" object
                                start_obj = data.get("start", {})
                                stream_sid = start_obj.get("streamSid") or data.get("streamSid") or data.get("stream_sid")
                                if stream_sid:
                                    logger.info(f"[Twilio] Stream SID: {stream_sid}")
                                    # Store streamSid in context for later use
                                    if context:
                                        context.stream_sid = stream_sid
                                else:
                                    logger.warning("[Twilio] No streamSid found in start event")
                                
                                if not extracted_phone_number:
                                    extracted_phone_number = data.get("phoneNumber") or data.get("phone_number")
                                # Extract Stream parameters if available
                                stream_params = data.get("streamParams", {})
                                if isinstance(stream_params, dict):
                                    if not scheduled_call_id:
                                        scheduled_call_id = stream_params.get("scheduled_call_id")
                                    if not user_name:
                                        user_name = stream_params.get("user_name")
                                
                                # Check for track information in start event
                                track = data.get("track", "unknown")
                                logger.info(f"[Twilio] Stream track configuration: {track}")
                                print(f"[Twilio] Stream track configuration: {track}")

                            elif event_type == "media":
                                # Media stream data - audio comes as base64-encoded payload in media events
                                # According to Twilio docs: https://www.twilio.com/docs/voice/tutorials/consume-real-time-media-stream-using-websockets-python-and-flask
                                # The payload is in data['media']['payload'], not data['payload']
                                
                                # Extract payload from data['media']['payload'] as per Twilio documentation
                                media_obj = data.get("media", {})
                                payload = media_obj.get("payload", "") if isinstance(media_obj, dict) else ""
                                
                                if payload:
                                    # Decode base64 payload to get audio bytes
                                    try:
                                        audio_bytes = base64.b64decode(payload)
                                        audio_size = len(audio_bytes)
                                        binary_message_count += 1  # Count this as audio data received
                                        
                                        # Only log occasionally to reduce spam (every 50th chunk)
                                        if binary_message_count % 50 == 0:
                                            logger.debug(f"[Twilio] Audio chunks received: {binary_message_count} ({audio_size} bytes each)")
                                        
                                        # Send to Deepgram if connection exists
                                        if deepgram_connection:
                                            await deepgram_service.send_audio(deepgram_connection, audio_bytes)
                                        else:
                                            logger.warning("[Twilio] Deepgram connection not initialized, skipping audio")
                                    except Exception as e:
                                        logger.error(f"[Twilio] Error decoding media payload: {e}")
                                        print(f"[Twilio] ❌ Error decoding media payload: {e}")
                                # Don't log empty media events - they're just heartbeats/metadata

                            elif event_type == "stop":
                                # Call ended
                                print(f"Call stopped: {extracted_phone_number}")
                                logger.info(f"[Twilio] Call ended. Stats - Binary messages: {binary_message_count}, Text messages: {text_message_count}, Media events: {media_event_count}")
                                print(f"[Twilio] 📊 Call stats - Binary audio: {binary_message_count}, Text messages: {text_message_count}, Media events: {media_event_count}")
                                break

                        except json.JSONDecodeError:
                            # Invalid JSON, ignore
                            pass
                    
                    # Periodic stats logging during the call (every 10 seconds)
                    current_time = time.time()
                    if current_time - last_stats_log >= stats_log_interval:
                        logger.info(f"[Twilio] 📊 Live stats - Binary: {binary_message_count}, Text: {text_message_count}, Media events: {media_event_count}")
                        print(f"[Twilio] 📊 Live stats - Binary audio: {binary_message_count}, Text messages: {text_message_count}, Media events: {media_event_count}")
                        last_stats_log = current_time

                    # Check if we have new transcribed text
                    # Only process if we have transcriptions and they're not too short
                    if transcribed_text_buffer:
                        # Process the latest transcription
                        transcribed_text = transcribed_text_buffer.pop(0)
                        
                        # Filter out very short transcriptions (likely partial words)
                        # Only process if transcription is meaningful (more than 2 characters)
                        if transcribed_text and len(transcribed_text.strip()) > 2:
                            # Process this utterance
                            await self._process_utterance(
                                transcribed_text=transcribed_text,
                                context=context,
                            )
                        else:
                            logger.debug(f"[Twilio] Skipping short transcription: '{transcribed_text}'")

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
            connection_key = extracted_call_sid or extracted_phone_number or "default"
            if connection_key and connection_key in self.active_connections:
                del self.active_connections[connection_key]

            # Close Deepgram connection
            if deepgram_connection:
                try:
                    await deepgram_service.close_connection(deepgram_connection)
                except Exception as e:
                    logger.warning(f"[Twilio] Error closing Deepgram connection: {e}")

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
            logger.info(f"[Processing] 🎯 Processing utterance: {transcribed_text}")
            print(f"[Processing] 🎯 Processing: {transcribed_text}")
            
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
            # Check if WebSocket is still open before sending
            websocket = context.websocket
            if not websocket:
                logger.warning("[TTS] WebSocket is None, cannot send audio")
                return
            
            # Check WebSocket state (FastAPI WebSocket uses client_state)
            try:
                if hasattr(websocket, "client_state"):
                    if websocket.client_state.name != "CONNECTED":
                        logger.warning(f"[TTS] WebSocket not connected (state: {websocket.client_state.name}), skipping TTS")
                        return
            except Exception:
                # If we can't check state, try to send anyway (will fail gracefully)
                pass
            
            # Convert text to speech
            audio_bytes = await tts_service.text_to_speech(text)

            if audio_bytes:
                # Convert audio to μ-law format for Twilio Media Streams
                # Twilio expects: μ-law (mulaw) PCM, 8000 Hz, mono
                mulaw_audio = await self._convert_to_mulaw(audio_bytes)
                
                if mulaw_audio:
                    # Send audio to Twilio via WebSocket as JSON messages with base64-encoded payload
                    # Twilio Media Streams requires JSON format, not binary
                    # Format: {"event": "media", "streamSid": "...", "media": {"payload": "<base64>"}}
                    
                    stream_sid = getattr(context, 'stream_sid', None)
                    if not stream_sid:
                        logger.warning("[TTS] No streamSid available, cannot send audio")
                        return
                    
                    # Send in chunks for real-time playback (160 bytes = 20ms at 8000Hz μ-law)
                    chunk_size = 160  # 20ms of audio at 8000Hz (8000 * 0.02 = 160 bytes for μ-law)
                    total_chunks = 0
                    
                    for i in range(0, len(mulaw_audio), chunk_size):
                        chunk = mulaw_audio[i:i + chunk_size]
                        if chunk:
                            # Base64 encode the chunk
                            encoded_payload = base64.b64encode(chunk).decode('utf-8')
                            
                            # Create JSON message
                            media_message = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": encoded_payload
                                }
                            }
                            
                            # Send as JSON text message
                            await websocket.send_text(json.dumps(media_message))
                            total_chunks += 1
                            # No delay needed - Twilio will buffer and play chunks smoothly
                    
                    logger.info(f"[TTS] ✅ Sent {len(mulaw_audio)} bytes of μ-law audio in {total_chunks} JSON messages to Twilio")
                else:
                    logger.warning("[TTS] Failed to convert audio to μ-law format")

        except Exception as e:
            # Don't log errors if WebSocket is already closed
            error_str = str(e)
            if "close message" not in error_str.lower() and "disconnect" not in error_str.lower():
                logger.error(f"[TTS] Error sending TTS response: {e}", exc_info=True)
            else:
                logger.debug(f"[TTS] WebSocket closed, skipping TTS: {e}")
    
    async def _convert_to_mulaw(self, audio_bytes: bytes) -> bytes:
        """
        Convert audio bytes to μ-law format for Twilio Media Streams.
        
        Args:
            audio_bytes: Audio bytes from TTS (MP3/WAV format)
            
        Returns:
            μ-law encoded audio bytes at 8000 Hz, mono
        """
        try:
            import io
            try:
                import audioop
            except ImportError:
                # Python 3.13+ removed audioop, use backport
                import audioop_lts as audioop
            from pydub import AudioSegment
            
            logger.debug(f"[TTS] Converting {len(audio_bytes)} bytes to μ-law format...")
            
            # Load audio from bytes (handles MP3, WAV, etc.)
            # pydub requires ffmpeg for MP3, but can handle WAV natively
            # ElevenLabs returns MP3 by default, so we need ffmpeg
            try:
                # Try to auto-detect format (requires ffmpeg for MP3)
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            except Exception as e:
                # If auto-detect fails (likely no ffmpeg), try WAV format
                logger.warning(f"[TTS] Auto-detect failed (may need ffmpeg): {e}. Trying WAV...")
                try:
                    audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
                except Exception as e2:
                    # If WAV also fails, try MP3 explicitly (requires ffmpeg)
                    logger.warning(f"[TTS] WAV format failed: {e2}. Trying MP3...")
                    try:
                        audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                    except Exception as e3:
                        logger.error(f"[TTS] All format attempts failed. Install ffmpeg: brew install ffmpeg (macOS)")
                        raise e3
            
            logger.debug(f"[TTS] Original audio: {audio.frame_rate}Hz, {audio.channels}ch, {audio.sample_width*8}bit")
            
            # Convert to required format: 8000 Hz, mono, 16-bit PCM
            audio = audio.set_frame_rate(8000)  # Resample to 8000 Hz
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_sample_width(2)  # 16-bit (2 bytes per sample)
            
            # Get raw PCM data
            raw_audio = audio.raw_data
            
            # Convert PCM to μ-law using audioop
            mulaw_audio = audioop.lin2ulaw(raw_audio, 2)  # 2 = 16-bit sample width
            
            logger.info(f"[TTS] ✅ Converted {len(audio_bytes)} bytes → {len(mulaw_audio)} bytes μ-law (8000Hz, mono)")
            return mulaw_audio
            
        except ImportError as e:
            logger.error(f"[TTS] Missing dependency: {e}. Install with: pip install pydub")
            logger.error("[TTS] For MP3 support, also install ffmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
            return None
        except Exception as e:
            logger.error(f"[TTS] Error converting to μ-law: {e}", exc_info=True)
            return None

    async def _end_call(self, phone_number: str):
        """End the call by closing WebSocket connection."""
        if phone_number in self.active_connections:
            websocket = self.active_connections[phone_number]
            await websocket.close(code=1000, reason="Call ended by tool")


# Global instance
twilio_websocket_handler = TwilioWebSocketHandler()

