"""
Deepgram service for real-time speech-to-text transcription using Flux.
"""
from typing import Optional, Callable
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV2SocketClientResponse
import asyncio
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class DeepgramService:
    """Service for Deepgram STT (Speech-to-Text) using Flux model."""

    def __init__(self):
        self.api_key = settings.deepgram_api_key
        self.client = AsyncDeepgramClient(api_key=self.api_key)

    async def create_live_transcription(
        self,
        on_transcript: Callable[[str], None],
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Create a live transcription connection using Deepgram Flux.

        Args:
            on_transcript: Callback function called when transcription is available
                          Signature: (text: str) -> None
            on_error: Optional callback for errors

        Returns:
            Live transcription connection object (V2SocketClient)
        """
        logger.info("[Deepgram] Creating Flux connection...")
        
        # Create connection using Deepgram SDK v2 API with Flux model
        # Twilio sends audio/x-mulaw (μ-law) format at 8000 Hz
        connection_manager = self.client.listen.v2.connect(
            model="flux-general-en",
            encoding="mulaw",  # Twilio's format
            sample_rate="8000",  # Twilio's sample rate
        )
        
        # Enter the async context manager to get the connection object
        connection = await connection_manager.__aenter__()
        
        logger.info(f"[Deepgram] Connection established: {type(connection)}")
        
        # Track the latest transcript for end-of-turn detection
        latest_transcript = ""
        last_interim_callback_time = 0  # Track when we last called interim callback
        INTERIM_CALLBACK_THROTTLE = 0.5  # Only call interim callback every 500ms
        
        # Define message handler function (as per Flux docs)
        def on_message(message: ListenV2SocketClientResponse) -> None:
            """Handle transcription message from Deepgram Flux."""
            nonlocal latest_transcript, last_interim_callback_time
            try:
                import time
                # Check message type - ListenV2TurnInfoEvent is the Flux turn info event
                message_type = type(message).__name__
                
                # Log message attributes on first EndOfTurn to understand structure
                if not hasattr(on_message, '_logged_attributes'):
                    attrs = [attr for attr in dir(message) if not attr.startswith('_')]
                    logger.info(f"[Deepgram] Message type: {message_type}, Attributes: {attrs}")
                    # Log all attribute values
                    for attr in attrs:
                        try:
                            value = getattr(message, attr, None)
                            if not callable(value):
                                logger.debug(f"[Deepgram]   {attr} = {value}")
                        except:
                            pass
                    on_message._logged_attributes = True
                
                # Check for transcript attribute (try multiple possible attribute names)
                transcript_value = None
                if hasattr(message, 'transcript'):
                    transcript_value = getattr(message, 'transcript', None)
                elif hasattr(message, 'text'):
                    transcript_value = getattr(message, 'text', None)
                elif hasattr(message, 'message'):
                    transcript_value = getattr(message, 'message', None)
                
                # Log message details for debugging
                logger.debug(f"[Deepgram] Message type: {message_type}, transcript_value: {transcript_value}")
                
                # Update latest transcript if this is a transcript message
                if transcript_value:
                    latest_transcript = transcript_value
                    logger.info(f"[Deepgram] 📝 Transcript update received: {latest_transcript}")
                
                # Check for EndOfTurn - ListenV2TurnInfoEvent has an 'event' attribute
                # Check various possible attributes for EndOfTurn detection
                is_end_of_turn = False
                
                # Method 1: Check if message has an 'event' attribute set to 'EndOfTurn'
                if hasattr(message, 'event'):
                    event_value = getattr(message, 'event', None)
                    if event_value == "EndOfTurn" or event_value == "end_of_turn":
                        is_end_of_turn = True
                        logger.info(f"[Deepgram] ✅ EndOfTurn detected via event attribute: {event_value}")
                
                # Method 2: Check if message type indicates EndOfTurn
                if "EndOfTurn" in message_type or "end_of_turn" in message_type.lower():
                    is_end_of_turn = True
                    logger.info(f"[Deepgram] ✅ EndOfTurn detected via message type: {message_type}")
                
                # Method 3: Check for is_final or similar flags
                if hasattr(message, 'is_final') and getattr(message, 'is_final', False):
                    is_end_of_turn = True
                    logger.info(f"[Deepgram] ✅ EndOfTurn detected via is_final flag")
                
                # Call callback for interim transcriptions to enable immediate TTS interruption
                # This allows cancelling TTS as soon as speech is detected, not just on EndOfTurn
                # Check if we have a transcript update (even if not EndOfTurn)
                if transcript_value and latest_transcript and len(latest_transcript.strip()) > 1:
                    # Check if this is an interim transcription (not EndOfTurn)
                    is_interim = not is_end_of_turn
                    
                    if is_interim:
                        # Throttle interim callbacks to avoid excessive calls
                        current_time = time.time()
                        if current_time - last_interim_callback_time >= INTERIM_CALLBACK_THROTTLE:
                            # For interim transcriptions, call callback to interrupt TTS
                            # This provides instant interruption as soon as user starts speaking
                            logger.info(f"[Deepgram] 🛑 INTERIM transcript detected (for interruption): {latest_transcript}")
                            last_interim_callback_time = current_time
                            try:
                                # Call with is_interim=True to indicate this is a partial transcription
                                if asyncio.iscoroutinefunction(on_transcript):
                                    # Check if callback accepts is_interim parameter
                                    import inspect
                                    sig = inspect.signature(on_transcript)
                                    if 'is_interim' in sig.parameters:
                                        asyncio.create_task(on_transcript(latest_transcript, is_interim=True))
                                    else:
                                        # Old callback signature - call it anyway to trigger cancellation
                                        asyncio.create_task(on_transcript(latest_transcript))
                                else:
                                    # Sync callback - try to call with is_interim if supported
                                    import inspect
                                    sig = inspect.signature(on_transcript)
                                    if 'is_interim' in sig.parameters:
                                        on_transcript(latest_transcript, is_interim=True)
                                    else:
                                        on_transcript(latest_transcript)
                            except Exception as e:
                                logger.warning(f"[Deepgram] Error calling interim callback: {e}")
                    else:
                        logger.debug(f"[Deepgram] Transcript update is EndOfTurn, will process later: {latest_transcript}")
                
                # Process final transcript on EndOfTurn
                if is_end_of_turn:
                    if latest_transcript and len(latest_transcript.strip()) > 2:  # Filter out very short fragments
                        logger.info(f"[Deepgram] ✅ EndOfTurn - Processing final transcript: {latest_transcript}")
                        # Call the callback with final transcribed text (is_interim=False)
                        try:
                            if asyncio.iscoroutinefunction(on_transcript):
                                import inspect
                                sig = inspect.signature(on_transcript)
                                if 'is_interim' in sig.parameters:
                                    asyncio.create_task(on_transcript(latest_transcript, is_interim=False))
                                else:
                                    asyncio.create_task(on_transcript(latest_transcript))
                            else:
                                import inspect
                                sig = inspect.signature(on_transcript)
                                if 'is_interim' in sig.parameters:
                                    on_transcript(latest_transcript, is_interim=False)
                                else:
                                    on_transcript(latest_transcript)
                        except Exception as e:
                            logger.warning(f"[Deepgram] Error calling final callback: {e}")
                        latest_transcript = ""  # Reset after processing
                    else:
                        logger.debug(f"[Deepgram] EndOfTurn but transcript too short, skipping: {latest_transcript}")
            except Exception as e:
                logger.error(f"[Deepgram] Error handling transcript: {e}", exc_info=True)
                if on_error:
                    if asyncio.iscoroutinefunction(on_error):
                        asyncio.create_task(on_error(e))
                    else:
                        on_error(e)
        
        def handle_error(error):
            """Handle errors from Deepgram."""
            logger.error(f"[Deepgram] Connection error: {error}")
            if on_error:
                if asyncio.iscoroutinefunction(on_error):
                    asyncio.create_task(on_error(error))
                else:
                    on_error(error)
        
        # Set up event handlers using EventType (as per Flux docs)
        connection.on(EventType.OPEN, lambda _: logger.info("[Deepgram] Connection opened"))
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.CLOSE, lambda _: logger.info("[Deepgram] Connection closed"))
        connection.on(EventType.ERROR, handle_error)
        
        # Start the connection listening in background (as per Flux docs)
        deepgram_task = asyncio.create_task(connection.start_listening())
        logger.info("[Deepgram] Event handlers registered and listening started")
        
        # Store the task and connection manager for cleanup
        connection._deepgram_task = deepgram_task
        connection._connection_manager = connection_manager
        
        return connection

    async def send_audio(self, connection, audio_data: bytes):
        """
        Send audio data to Deepgram for transcription.

        Args:
            connection: Live transcription connection (V2SocketClient)
            audio_data: Audio bytes to transcribe
        """
        try:
            # Send audio data using _send() method (as per Flux docs)
            # For v2 API, use _send() method
            if hasattr(connection, "_send"):
                await connection._send(audio_data)
                logger.debug(f"[Deepgram] Sent {len(audio_data)} bytes via _send()")
            elif hasattr(connection, "send_media"):
                connection.send_media(audio_data)
                logger.debug(f"[Deepgram] Sent {len(audio_data)} bytes via send_media()")
            elif hasattr(connection, "send"):
                connection.send(audio_data)
                logger.debug(f"[Deepgram] Sent {len(audio_data)} bytes via send()")
            else:
                logger.warning(f"[Deepgram] No send method found. Available: {[m for m in dir(connection) if 'send' in m.lower()]}")
        except Exception as e:
            logger.error(f"[Deepgram] Error sending audio: {e}", exc_info=True)
            raise

    async def close_connection(self, connection):
        """Close Deepgram connection."""
        try:
            # Cancel the listening task if it exists
            if hasattr(connection, "_deepgram_task"):
                connection._deepgram_task.cancel()
                try:
                    await connection._deepgram_task
                except asyncio.CancelledError:
                    pass
            
            # Exit the async context manager if it exists
            if hasattr(connection, "_connection_manager"):
                await connection._connection_manager.__aexit__(None, None, None)
            
            # Try other close methods
            if hasattr(connection, "finish"):
                connection.finish()
            elif hasattr(connection, "close"):
                connection.close()
        except Exception as e:
            logger.warning(f"[Deepgram] Error closing connection: {e}")


# Global instance
deepgram_service = DeepgramService()
