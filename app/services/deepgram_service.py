"""
Deepgram service for real-time speech-to-text transcription.
"""
from typing import Optional, Callable, AsyncGenerator
from deepgram import DeepgramClient
import asyncio

from app.core.config import settings


class DeepgramService:
    """Service for Deepgram STT (Speech-to-Text)."""

    def __init__(self):
        self.api_key = settings.deepgram_api_key
        self.client = DeepgramClient(api_key=self.api_key)

    async def create_live_transcription(
        self,
        on_transcript: Callable[[str], None],
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Create a live transcription connection.

        Args:
            on_transcript: Callback function called when transcription is available
                          Signature: (text: str) -> None
            on_error: Optional callback for errors

        Returns:
            Live transcription connection object (V1SocketClient)
        """
        # Create connection context manager using Deepgram SDK v1 API
        connection_manager = self.client.listen.v1.connect(
            model="nova-2",
            language="en-US",
            smart_format="true",
            encoding="linear16",
            channels="1",
            sample_rate="8000",
            interim_results="false",  # Only final results
        )

        # Enter the context manager to get the connection object
        connection = connection_manager.__enter__()

        # Set up event handlers - check for available methods
        # The V1SocketClient API varies, so we'll handle messages in a background task
        async def message_handler():
            """Handle messages from Deepgram."""
            try:
                # Check if connection has a listen/read method
                if hasattr(connection, "listen"):
                    async for message in connection.listen():
                        await self._handle_message(message, on_transcript, on_error)
                elif hasattr(connection, "read"):
                    while True:
                        message = await connection.read()
                        if message:
                            await self._handle_message(message, on_transcript, on_error)
            except Exception as e:
                if on_error:
                    if asyncio.iscoroutinefunction(on_error):
                        await on_error(e)
                    else:
                        on_error(e)

        # Start message handler in background
        asyncio.create_task(message_handler())

        return connection

    async def _handle_message(self, message, on_transcript, on_error):
        """Handle a message from Deepgram."""
        try:
            # Parse the transcript from message
            # The message structure may vary - adjust based on actual API
            if hasattr(message, "channel") and message.channel:
                if hasattr(message.channel, "alternatives") and message.channel.alternatives:
                    transcript = message.channel.alternatives[0].transcript
                    if transcript:
                        # Call the callback with transcribed text
                        if asyncio.iscoroutinefunction(on_transcript):
                            await on_transcript(transcript)
                        else:
                            on_transcript(transcript)
        except Exception as e:
            if on_error:
                if asyncio.iscoroutinefunction(on_error):
                    await on_error(e)
                else:
                    on_error(e)

    async def send_audio(self, connection, audio_data: bytes):
        """
        Send audio data to Deepgram for transcription.

        Args:
            connection: Live transcription connection (V1SocketClient)
            audio_data: Audio bytes to transcribe
        """
        # Send audio data through connection
        if hasattr(connection, "send"):
            connection.send(audio_data)
        elif hasattr(connection, "write"):
            connection.write(audio_data)

    def close_connection(self, connection):
        """Close Deepgram connection."""
        try:
            if hasattr(connection, "finish"):
                connection.finish()
            elif hasattr(connection, "close"):
                connection.close()
            # Exit context manager if it's still active
            if hasattr(connection, "__exit__"):
                connection.__exit__(None, None, None)
        except:
            pass


# Global instance
deepgram_service = DeepgramService()
