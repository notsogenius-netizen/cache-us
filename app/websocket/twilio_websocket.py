"""
Twilio WebSocket handler for Media Streams.
"""
import json
import logging
import asyncio
import base64
import time
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

            async def on_transcript(text: str, is_interim: bool = False):
                """Handle Deepgram transcription.
                
                Args:
                    text: Transcribed text
                    is_interim: Whether this is an interim (partial) transcription
                """
                nonlocal current_transcription
                current_transcription = text
                
                # Cancel any ongoing TTS immediately when user starts speaking
                # This provides instant interruption without waiting for processing
                if context and context.active_tts_task and not context.active_tts_task.done():
                    # GRACE PERIOD: Don't cancel TTS for first 0.5 seconds after it starts
                    # This prevents false positives from echo/feedback
                    if hasattr(context, '_tts_start_time'):
                        time_since_tts_start = time.time() - context._tts_start_time
                        grace_period = 0.5  # 0.5 second grace period for Deepgram (shorter since it's more accurate)
                        if time_since_tts_start < grace_period:
                            logger.debug(f"[Deepgram] ⏱️ Ignoring speech during grace period ({time_since_tts_start:.2f}s < {grace_period}s)")
                            return
                    
                    logger.info(f"[Deepgram] 🛑 User speech detected (interim={is_interim}) - cancelling TTS immediately")
                    print(f"[Deepgram] 🛑 Cancelling TTS due to user speech...")
                    
                    # Cancel TTS immediately - this needs to happen FAST
                    try:
                        # Set cancellation event FIRST for immediate effect in the chunk loop
                        if context.tts_cancellation_event is None:
                            context.tts_cancellation_event = asyncio.Event()
                        context.tts_cancellation_event.set()
                        
                        # Cancel the task immediately (this will raise CancelledError in the task)
                        context.active_tts_task.cancel()
                        
                        # Send clear marker in background to clear Twilio buffer
                        # This doesn't block the callback
                        websocket = context.websocket
                        stream_sid = getattr(context, 'stream_sid', None)
                        if websocket and stream_sid:
                            asyncio.create_task(self._send_clear_marker(websocket, stream_sid, context))
                        
                        logger.info(f"[Deepgram] ✅ TTS cancellation signal sent (task cancelled, event set)")
                    except Exception as cancel_error:
                        logger.warning(f"[Deepgram] Error cancelling TTS: {cancel_error}")
                
                # Only add to buffer if it's a final transcription (EndOfTurn)
                # Interim transcriptions are used for interruption only
                if not is_interim:
                    transcribed_text_buffer.append(text)
                    logger.info(f"[Deepgram] 📝 Final transcription received: {text}")
                    print(f"[Deepgram] 📝 Final transcription: {text}")
                else:
                    logger.debug(f"[Deepgram] 📝 Interim transcription (interrupting TTS): {text}")

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
                        
                        # Check if user is speaking (receiving audio) while TTS is active
                        # Cancel TTS immediately when we detect incoming audio with voice activity
                        # Use a simple voice activity detection: check for audio variation
                        if context and context.active_tts_task and not context.active_tts_task.done() and audio_size > 0:
                            # Initialize voice activity tracking if not exists
                            if not hasattr(context, '_voice_activity_chunks'):
                                context._voice_activity_chunks = 0
                                context._last_voice_check_time = time.time()
                            
                            # GRACE PERIOD: Don't check for voice activity for first 1 second after TTS starts
                            # This prevents false positives from echo/feedback
                            if not hasattr(context, '_tts_start_time'):
                                context._tts_start_time = time.time()
                            
                            time_since_tts_start = time.time() - context._tts_start_time
                            grace_period = 1.0  # 1 second grace period
                            
                            if time_since_tts_start < grace_period:
                                # Still in grace period - ignore voice activity
                                logger.debug(f"[Twilio] ⏱️ Grace period active ({time_since_tts_start:.2f}s < {grace_period}s) - ignoring voice activity")
                                continue
                            
                            # Check if audio contains actual voice (not just silence or echo)
                            # μ-law silence is typically 0xFF or 0x00, check for variation
                            audio_variation = max(audio_bytes) - min(audio_bytes) if audio_bytes else 0
                            # Also check for non-silence bytes (not all 0xFF or 0x00)
                            non_silence_ratio = sum(1 for b in audio_bytes if b != 0xFF and b != 0x00) / len(audio_bytes) if audio_bytes else 0
                            
                            # INCREASED THRESHOLDS: More strict to avoid false positives
                            # Voice activity threshold: variation > 30 AND at least 40% non-silence
                            # (Increased from 15/20% to 30/40% to be less sensitive)
                            if audio_variation > 30 and non_silence_ratio > 0.4:
                                context._voice_activity_chunks += 1
                                # Require sustained voice activity (at least 3 chunks ~60ms) to avoid false positives
                                if context._voice_activity_chunks >= 3:
                                    logger.info(f"[Twilio] 🛑 Sustained voice activity detected (chunks={context._voice_activity_chunks}, variation={audio_variation}, non_silence={non_silence_ratio:.2%}, time_since_tts={time_since_tts_start:.2f}s) - cancelling TTS")
                                    print(f"[Twilio] 🛑 Sustained voice activity detected (chunks={context._voice_activity_chunks}, variation={audio_variation}, non_silence={non_silence_ratio:.2%}, time_since_tts={time_since_tts_start:.2f}s) - cancelling TTS")
                                    logger.info(f"[Twilio] 📊 Voice activity stats: audio_size={audio_size}, max={max(audio_bytes) if audio_bytes else 0}, min={min(audio_bytes) if audio_bytes else 0}")
                                    print(f"[Twilio] 📊 Voice activity stats: audio_size={audio_size}, max={max(audio_bytes) if audio_bytes else 0}, min={min(audio_bytes) if audio_bytes else 0}")
                                    # Set cancellation event immediately
                                    if context.tts_cancellation_event is None:
                                        context.tts_cancellation_event = asyncio.Event()
                                    context.tts_cancellation_event.set()
                                    # Cancel task and send clear marker
                                    context.active_tts_task.cancel()
                                    stream_sid = getattr(context, 'stream_sid', None)
                                    if stream_sid:
                                        asyncio.create_task(self._send_clear_marker(websocket, stream_sid, context))
                                    # Reset voice activity tracking
                                    context._voice_activity_chunks = 0
                            else:
                                # Reset counter if no voice activity
                                context._voice_activity_chunks = 0
                        
                        # Only log first few binary messages to avoid spam
                        if binary_message_count <= 3:
                            logger.info(f"[Twilio] 🔊 Received binary audio data: {audio_size} bytes")
                            print(f"[Twilio] 🔊 Received binary audio data: {audio_size} bytes")
                            
                            # Log first few bytes for debugging (first 10 bytes as hex)
                            if audio_size > 0:
                                hex_preview = " ".join(f"{b:02x}" for b in audio_bytes[:10])
                                logger.debug(f"[Twilio] Audio data preview (first 10 bytes): {hex_preview}")
                        
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
                                        
                                        # Check if user is speaking (receiving audio) while TTS is active
                                        # Cancel TTS immediately when we detect incoming audio with voice activity
                                        if context and context.active_tts_task and not context.active_tts_task.done() and audio_size > 0:
                                            # Initialize voice activity tracking if not exists
                                            if not hasattr(context, '_voice_activity_chunks'):
                                                context._voice_activity_chunks = 0
                                            
                                            # GRACE PERIOD: Don't check for voice activity for first 1 second after TTS starts
                                            # This prevents false positives from echo/feedback
                                            if not hasattr(context, '_tts_start_time'):
                                                context._tts_start_time = time.time()
                                            
                                            time_since_tts_start = time.time() - context._tts_start_time
                                            grace_period = 1.0  # 1 second grace period
                                            
                                            if time_since_tts_start < grace_period:
                                                # Still in grace period - ignore voice activity
                                                logger.debug(f"[Twilio] ⏱️ Grace period active ({time_since_tts_start:.2f}s < {grace_period}s) - ignoring voice activity")
                                                continue
                                            
                                            # Check if audio contains actual voice (not just silence or echo)
                                            audio_variation = max(audio_bytes) - min(audio_bytes) if audio_bytes else 0
                                            non_silence_ratio = sum(1 for b in audio_bytes if b != 0xFF and b != 0x00) / len(audio_bytes) if audio_bytes else 0
                                            
                                            # INCREASED THRESHOLDS: More strict to avoid false positives
                                            # Voice activity threshold: variation > 30 AND at least 40% non-silence
                                            # (Increased from 15/20% to 30/40% to be less sensitive)
                                            if audio_variation > 30 and non_silence_ratio > 0.4:
                                                context._voice_activity_chunks += 1
                                                # Require sustained voice activity (at least 3 chunks ~60ms) to avoid false positives
                                                if context._voice_activity_chunks >= 3:
                                                    logger.info(f"[Twilio] 🛑 Sustained voice activity detected (chunks={context._voice_activity_chunks}, variation={audio_variation}, non_silence={non_silence_ratio:.2%}, time_since_tts={time_since_tts_start:.2f}s) - cancelling TTS")
                                                    print(f"[Twilio] 🛑 Sustained voice activity detected (chunks={context._voice_activity_chunks}, variation={audio_variation}, non_silence={non_silence_ratio:.2%}, time_since_tts={time_since_tts_start:.2f}s) - cancelling TTS")
                                                    # Set cancellation event immediately
                                                    if context.tts_cancellation_event is None:
                                                        context.tts_cancellation_event = asyncio.Event()
                                                    context.tts_cancellation_event.set()
                                                    # Cancel task and send clear marker
                                                    context.active_tts_task.cancel()
                                                    stream_sid = getattr(context, 'stream_sid', None)
                                                    if stream_sid:
                                                        asyncio.create_task(self._send_clear_marker(websocket, stream_sid, context))
                                                    # Reset voice activity tracking
                                                    context._voice_activity_chunks = 0
                                            else:
                                                # Reset counter if no voice activity
                                                context._voice_activity_chunks = 0
                                        
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

            # Cancel any ongoing TTS before closing
            if 'context' in locals() and context:
                try:
                    await self._cancel_ongoing_tts(context)
                except Exception as e:
                    logger.warning(f"[Twilio] Error cancelling TTS during cleanup: {e}")

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
        This method will cancel any ongoing TTS before processing new utterance.

        Args:
            transcribed_text: User's transcribed text
            context: ConversationContext
        """
        try:
            logger.info(f"[Processing] 🎯 Processing utterance: {transcribed_text}")
            print(f"[Processing] 🎯 Processing: {transcribed_text}")
            
            # Cancel any ongoing TTS before processing new utterance
            await self._cancel_ongoing_tts(context)
            
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

            # Log LLM raw response
            logger.info(f"[LLM] 📤 Raw LLM response: type={llm_response.get('type')}, content={llm_response.get('content', '')[:200]}...")
            print(f"[LLM] 📤 Raw LLM response: type={llm_response.get('type')}, content={llm_response.get('content', '')[:200]}...")

            # Handle LLM response
            final_response_text = await self._handle_llm_response(
                llm_response=llm_response,
                context=context,
            )

            # Log final response text that will be converted to TTS
            if final_response_text:
                logger.info(f"[LLM] 🎤 Text to convert to TTS ({len(final_response_text)} chars): {final_response_text[:300]}...")
                print(f"[LLM] 🎤 Text to convert to TTS ({len(final_response_text)} chars): {final_response_text[:300]}...")
            else:
                logger.warning(f"[LLM] ⚠️ No final response text - TTS will not be generated")
                print(f"[LLM] ⚠️ No final response text - TTS will not be generated")

            if final_response_text:
                # Convert to speech and send (as a tracked task)
                # Create task with cleanup wrapper
                async def send_tts_with_cleanup():
                    """Wrapper to ensure TTS task is cleaned up when done."""
                    try:
                        await self._send_tts_response(final_response_text, context)
                    finally:
                        # Clear task reference when done (only if still set to this task)
                        # Use asyncio.current_task() to get the current task
                        current_task = asyncio.current_task()
                        if context.active_tts_task is current_task:
                            context.active_tts_task = None
                
                # Create and track the TTS task
                tts_task = asyncio.create_task(send_tts_with_cleanup())
                context.active_tts_task = tts_task
                
                # Don't await - let it run in background so we can interrupt it
                # The task will be cleaned up when it completes or is cancelled

        except Exception as e:
            print(f"Error processing utterance: {e}")
            logger.error(f"[Processing] Error processing utterance: {e}", exc_info=True)
            # Send error message via TTS
            error_message = "I apologize, but I encountered an error processing your request."
            
            async def send_error_tts_with_cleanup():
                """Wrapper to ensure error TTS task is cleaned up when done."""
                try:
                    await self._send_tts_response(error_message, context)
                finally:
                    # Clear task reference when done (only if still set to this task)
                    current_task = asyncio.current_task()
                    if context.active_tts_task is current_task:
                        context.active_tts_task = None
            
            error_tts_task = asyncio.create_task(send_error_tts_with_cleanup())
            context.active_tts_task = error_tts_task
    
    async def _cancel_ongoing_tts(self, context):
        """
        Cancel any ongoing TTS operation for the given context.
        This also sends a marker to Twilio to clear the audio buffer.

        Args:
            context: ConversationContext
        """
        try:
            # Cancel any active TTS task
            if context.active_tts_task and not context.active_tts_task.done():
                logger.info("[TTS] ⚠️ Cancelling ongoing TTS task due to new utterance")
                print("[TTS] ⚠️ Cancelling ongoing TTS...")
                
                # Set cancellation event FIRST - this will stop the chunk loop immediately
                if context.tts_cancellation_event:
                    context.tts_cancellation_event.set()
                
                # Send marker to clear Twilio buffer immediately
                websocket = context.websocket
                stream_sid = getattr(context, 'stream_sid', None)
                if websocket and stream_sid:
                    try:
                        await self._send_clear_marker(websocket, stream_sid, context)
                    except Exception as marker_error:
                        logger.warning(f"[TTS] Error sending clear marker: {marker_error}")
                
                # Cancel the task
                context.active_tts_task.cancel()
                
                # Wait for cancellation to complete (with shorter timeout for faster response)
                try:
                    await asyncio.wait_for(context.active_tts_task, timeout=0.1)
                except asyncio.TimeoutError:
                    logger.debug("[TTS] TTS task cancellation timeout (task may continue in background)")
                except asyncio.CancelledError:
                    # Task was successfully cancelled
                    logger.info("[TTS] ✅ TTS task cancelled successfully")
                    pass
                except Exception as e:
                    logger.warning(f"[TTS] Error during TTS cancellation: {e}")
                
                # Clear the task reference
                context.active_tts_task = None
            
            # Reset cancellation event for new TTS operation
            if context.tts_cancellation_event:
                context.tts_cancellation_event.clear()
                
        except Exception as e:
            logger.error(f"[TTS] Error cancelling TTS: {e}", exc_info=True)
    
    async def _send_clear_marker(self, websocket, stream_sid: str, context):
        """
        Send a marker to Twilio to clear the audio buffer.
        This helps stop playback of buffered audio immediately.

        Args:
            websocket: WebSocket connection
            stream_sid: Twilio stream SID
            context: ConversationContext (for logging)
        """
        try:
            # Send a marker event to clear the buffer
            # According to Twilio docs, markers can be used to track playback
            # We'll use a unique marker name to signal buffer clearing
            import uuid
            marker_name = f"clear_{uuid.uuid4().hex[:8]}"
            
            marker_message = {
                "event": "mark",
                "streamSid": stream_sid,
                "mark": {
                    "name": marker_name
                }
            }
            
            await websocket.send_text(json.dumps(marker_message))
            logger.info(f"[TTS] 📍 Sent clear marker to Twilio: {marker_name}")
            
            # Also send a few empty/silence chunks to help clear the buffer
            # Twilio may continue playing buffered audio, so we send silence
            # Generate 20ms of silence (160 bytes of μ-law silence = 0xFF)
            silence_chunk = bytes([0xFF] * 160)
            
            # Send 5 silence chunks (100ms total) to help flush buffer
            for _ in range(5):
                encoded_payload = base64.b64encode(silence_chunk).decode('utf-8')
                silence_message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": encoded_payload
                    }
                }
                await websocket.send_text(json.dumps(silence_message))
            
            logger.debug("[TTS] Sent silence chunks to help clear buffer")
            
        except Exception as e:
            logger.warning(f"[TTS] Error sending clear marker: {e}")

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
            logger.info(f"[LLM] 📝 Text response received: {text[:200]}...")
            print(f"[LLM] 📝 Text response received: {text[:200]}...")
            if text:
                context.add_assistant_message(text)
            return text

        elif response_type == "tool_call":
            # LLM wants to call a tool
            tool_name = llm_response.get("tool_name")
            tool_arguments = llm_response.get("tool_arguments", {})
            
            logger.info(f"[Twilio] 🔧 LLM requested tool call: {tool_name} with args: {tool_arguments}")
            print(f"[Twilio] 🔧 LLM requested tool call: {tool_name} with args: {tool_arguments}")
            
            # IMPORTANT: Add the assistant's tool call message to conversation history
            # This is required for Cerebras API to match the tool_call_id
            assistant_message = llm_response.get("assistant_message")
            if assistant_message:
                # Add assistant's message with tool call to conversation history
                context.conversation_history.append(assistant_message)

            # Find tool definition
            tool_definition = None
            logger.info(f"[Twilio] 🔍 Looking for tool: {tool_name} in {len(context.tools)} available tools")
            print(f"[Twilio] 🔍 Looking for tool: {tool_name} in {len(context.tools)} available tools")
            for tool in context.tools:
                logger.info(f"[Twilio] 📋 Available tool: {tool.tool_name} (ID: {tool.tool_id})")
                print(f"[Twilio] 📋 Available tool: {tool.tool_name} (ID: {tool.tool_id})")
                if tool.tool_name == tool_name:
                    tool_definition = tool
                    logger.info(f"[Twilio] ✅ Found tool definition: {tool.tool_name}, parameters: {tool.parameters}")
                    print(f"[Twilio] ✅ Found tool definition: {tool.tool_name}, parameters: {tool.parameters}")
                    break

            if not tool_definition:
                logger.error(f"[Twilio] ❌ Tool '{tool_name}' not found in available tools")
                print(f"[Twilio] ❌ Tool '{tool_name}' not found in available tools")
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
            # Get tool_call_id from the original LLM response if available
            tool_call_id = llm_response.get("tool_call_id", "call_1")
            llm_final_response = await llm_service.generate_response_with_tool_result(
                agent_prompt=context.prompt,
                conversation_history=context.conversation_history,
                tool_name=tool_name,
                tool_result=tool_result_str,
                tools=context.tools,
                tool_call_id=tool_call_id,
            )

            # Handle final response (could be text or another tool call)
            return await self._handle_llm_response(llm_final_response, context)

        return None

    async def _send_tts_response(self, text: str, context):
        """
        Convert text to speech and send audio to client.
        This method is cancellable - will stop sending chunks if interrupted.

        Args:
            text: Text to convert to speech
            context: ConversationContext
        """
        try:
            # Initialize cancellation event if it doesn't exist
            if context.tts_cancellation_event is None:
                context.tts_cancellation_event = asyncio.Event()
            else:
                # Reset the event for new TTS operation
                context.tts_cancellation_event.clear()
            
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
            
            # Check for cancellation before generating TTS (user might have interrupted already)
            if context.tts_cancellation_event.is_set():
                logger.info("[TTS] TTS cancelled before generation")
                return
            
            # Convert text to speech
            logger.info(f"[TTS] 🎙️ Converting to speech: {text[:200]}...")
            print(f"[TTS] 🎙️ Converting to speech: {text[:200]}...")
            
            # Mark TTS start time for grace period
            if context:
                context._tts_start_time = time.time()
                logger.info(f"[TTS] ⏱️ TTS start time recorded - grace period active for 1 second")
                print(f"[TTS] ⏱️ TTS start time recorded - grace period active for 1 second")
            
            audio_bytes = await tts_service.text_to_speech(text)
            
            # Check for cancellation after TTS generation
            if context.tts_cancellation_event.is_set():
                logger.info("[TTS] TTS cancelled after generation")
                return

            if audio_bytes:
                # Convert audio to μ-law format for Twilio Media Streams
                # Twilio expects: μ-law (mulaw) PCM, 8000 Hz, mono
                mulaw_audio = await self._convert_to_mulaw(audio_bytes)
                
                # Check for cancellation after audio conversion
                if context.tts_cancellation_event.is_set():
                    logger.info("[TTS] TTS cancelled after audio conversion")
                    return
                
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
                        # Check for cancellation BEFORE processing chunk (most frequent check)
                        if context.tts_cancellation_event.is_set():
                            logger.info(f"[TTS] TTS interrupted after {total_chunks} chunks")
                            # Send marker to clear Twilio buffer
                            await self._send_clear_marker(websocket, stream_sid, context)
                            break
                        
                        # Yield to event loop to allow cancellation to be processed BEFORE sending
                        await asyncio.sleep(0)
                        
                        # Check again after yielding (cancellation might have happened during yield)
                        if context.tts_cancellation_event.is_set():
                            logger.info(f"[TTS] TTS interrupted after {total_chunks} chunks (after yield)")
                            await self._send_clear_marker(websocket, stream_sid, context)
                            break
                        
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
                            try:
                                # Check one more time right before the potentially blocking send
                                if context.tts_cancellation_event.is_set():
                                    logger.info(f"[TTS] TTS interrupted right before sending chunk")
                                    await self._send_clear_marker(websocket, stream_sid, context)
                                    break
                                
                                await websocket.send_text(json.dumps(media_message))
                                total_chunks += 1
                            except Exception as send_error:
                                # If send fails (connection closed), stop sending
                                logger.warning(f"[TTS] Error sending chunk: {send_error}")
                                break
                    
                    if not context.tts_cancellation_event.is_set():
                        logger.info(f"[TTS] ✅ Sent {len(mulaw_audio)} bytes of μ-law audio in {total_chunks} JSON messages to Twilio")
                    else:
                        logger.info(f"[TTS] ⚠️ TTS interrupted - sent {total_chunks} chunks before cancellation")
                else:
                    logger.warning("[TTS] Failed to convert audio to μ-law format")

        except asyncio.CancelledError:
            # Handle task cancellation gracefully
            logger.info("[TTS] TTS task cancelled")
            if context.tts_cancellation_event:
                context.tts_cancellation_event.set()
            raise
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

