"""
TTS service for ElevenLabs text-to-speech conversion.
"""
from typing import Optional
from elevenlabs import ElevenLabs
import asyncio

from app.core.config import settings


class TTSService:
    """Service for TTS (Text-to-Speech) using ElevenLabs."""

    def __init__(self):
        self.api_key = settings.elevenlabs_api_key
        self.voice_id = settings.elevenlabs_voice_id
        self.client = ElevenLabs(api_key=self.api_key)

    async def text_to_speech(
        self, text: str, voice_id: Optional[str] = None
    ) -> bytes:
        """
        Convert text to speech audio bytes.

        Args:
            text: Text to convert to speech
            voice_id: Optional voice ID (uses default from config if not provided)

        Returns:
            Audio bytes (WAV format)
        """
        try:
            voice_id_to_use = voice_id or self.voice_id or "default"
            
            # Generate audio using ElevenLabs
            # Use the text_to_speech API
            # ElevenLabs returns MP3 by default
            response = self.client.text_to_speech.convert(
                voice_id=voice_id_to_use,
                text=text,
                model_id="eleven_turbo_v2_5",  # or "eleven_multilingual_v2"
                optimize_streaming_latency=3,
            )

            # Collect audio bytes from response
            audio_bytes = b""
            for chunk in response:
                if hasattr(chunk, "data"):
                    audio_bytes += chunk.data
                elif isinstance(chunk, bytes):
                    audio_bytes += chunk
                else:
                    # If it's a generator or iterable, collect all chunks
                    try:
                        audio_bytes += b"".join(chunk) if hasattr(chunk, "__iter__") else bytes(chunk)
                    except:
                        pass
            
            return audio_bytes

        except Exception as e:
            # Log error
            print(f"Error in TTS service: {e}")
            # Return empty bytes on error
            return b""

    async def text_to_speech_streaming(
        self, text: str, voice_id: Optional[str] = None
    ) -> bytes:
        """
        Convert text to speech with streaming (for large texts).

        Args:
            text: Text to convert to speech
            voice_id: Optional voice ID

        Returns:
            Audio bytes
        """
        # For now, use the same implementation as text_to_speech
        # You can enhance this to return a generator/stream if needed
        return await self.text_to_speech(text, voice_id)


# Global instance
tts_service = TTSService()

