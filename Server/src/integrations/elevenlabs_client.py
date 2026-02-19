"""
ElevenLabs API Client for Audio Generation

This client provides async access to ElevenLabs APIs for:
- Text-to-Speech (TTS) generation
- Sound Effects (SFX) generation

API Documentation: https://elevenlabs.io/docs/api-reference

Environment Variables:
    ELEVENLABS_API_KEY: Your ElevenLabs API key

Usage:
    from integrations.elevenlabs_client import ElevenLabsClient

    client = ElevenLabsClient()
    
    # Generate speech
    result = await client.text_to_speech(
        text="Hello, welcome to the game!",
        voice_id="JBFqnCBsd6RMkjVDRZzb"  # George voice
    )
    
    # Generate sound effect
    result = await client.generate_sound_effect(
        text="Sword slash with metallic ring",
        duration_seconds=2.0
    )
"""

import os
import asyncio
import logging
from typing import Optional, Literal
from dataclasses import dataclass
from enum import Enum
import httpx

logger = logging.getLogger("adjoint-server.elevenlabs")


class ElevenLabsError(Exception):
    """Base exception for ElevenLabs API errors."""
    pass


class ElevenLabsAPIError(ElevenLabsError):
    """Raised when the ElevenLabs API returns an error."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"ElevenLabs API error ({status_code}): {message}")


class ElevenLabsConfigError(ElevenLabsError):
    """Raised when configuration is missing or invalid."""
    pass


class OutputFormat(str, Enum):
    """Available output formats for audio generation."""
    # MP3 formats
    MP3_22050_32 = "mp3_22050_32"
    MP3_44100_32 = "mp3_44100_32"
    MP3_44100_64 = "mp3_44100_64"
    MP3_44100_96 = "mp3_44100_96"
    MP3_44100_128 = "mp3_44100_128"
    MP3_44100_192 = "mp3_44100_192"  # Requires Creator tier
    
    # PCM formats
    PCM_16000 = "pcm_16000"
    PCM_22050 = "pcm_22050"
    PCM_24000 = "pcm_24000"
    PCM_44100 = "pcm_44100"  # Requires Pro tier
    
    # u-law format (for Twilio)
    ULAW_8000 = "ulaw_8000"


class TTSModel(str, Enum):
    """Available TTS models."""
    ELEVEN_MULTILINGUAL_V2 = "eleven_multilingual_v2"
    ELEVEN_TURBO_V2_5 = "eleven_turbo_v2_5"
    ELEVEN_TURBO_V2 = "eleven_turbo_v2"
    ELEVEN_MONOLINGUAL_V1 = "eleven_monolingual_v1"
    ELEVEN_FLASH_V2 = "eleven_flash_v2"
    ELEVEN_FLASH_V2_5 = "eleven_flash_v2_5"


class SFXModel(str, Enum):
    """Available sound effects models."""
    ELEVEN_TEXT_TO_SOUND_V2 = "eleven_text_to_sound_v2"


@dataclass
class VoiceSettings:
    """Voice settings for TTS generation."""
    stability: float = 0.5  # 0.0 to 1.0
    similarity_boost: float = 0.75  # 0.0 to 1.0
    style: float = 0.0  # 0.0 to 1.0
    use_speaker_boost: bool = True


@dataclass
class AudioResult:
    """Result of an audio generation request."""
    audio_data: bytes
    format: str
    character_cost: Optional[int] = None
    request_id: Optional[str] = None
    
    def save(self, filepath: str) -> None:
        """Save audio data to a file."""
        with open(filepath, 'wb') as f:
            f.write(self.audio_data)


@dataclass
class Voice:
    """Represents an ElevenLabs voice."""
    voice_id: str
    name: str
    category: str
    description: Optional[str] = None
    labels: Optional[dict] = None
    preview_url: Optional[str] = None


# Common pre-made voice IDs for convenience
class PreMadeVoices:
    """Pre-made voice IDs available in ElevenLabs."""
    RACHEL = "21m00Tcm4TlvDq8ikWAM"  # Female, calm
    DOMI = "AZnzlk1XvdvUeBnXmlld"    # Female, strong
    BELLA = "EXAVITQu4vr4xnSDxMaL"   # Female, soft
    ANTONI = "ErXwobaYiN019PkySvjV"  # Male, well-rounded
    ELLI = "MF3mGyEYCl7XYWbV9V6O"    # Female, emotional
    JOSH = "TxGEqnHWrfWFTfGW9XjX"    # Male, deep
    ARNOLD = "VR6AewLTigWG4xSOukaG"  # Male, crisp
    ADAM = "pNInz6obpgDQGcFmaJgB"    # Male, deep
    SAM = "yoZ06aMxZJJ28mfd3POQ"     # Male, raspy
    GEORGE = "JBFqnCBsd6RMkjVDRZzb"  # Male, British
    CHARLIE = "IKne3meq5aSn9XLyUdCD" # Male, casual
    EMILY = "LcfcDJNUP1GQjkzn1xUU"   # Female, British


class ElevenLabsClient:
    """
    Async client for ElevenLabs API.
    
    Provides methods for:
    - Text-to-Speech generation
    - Sound Effects generation
    - Voice listing and management
    """
    
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ElevenLabs client.
        
        Args:
            api_key: ElevenLabs API key. If not provided, reads from 
                    ELEVENLABS_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ElevenLabsConfigError(
                "ElevenLabs API key not provided. Set ELEVENLABS_API_KEY environment variable "
                "or pass api_key to constructor."
            )
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    # =========================================================================
    # Text-to-Speech
    # =========================================================================
    
    async def text_to_speech(
        self,
        text: str,
        voice_id: str = PreMadeVoices.GEORGE,
        model_id: TTSModel = TTSModel.ELEVEN_MULTILINGUAL_V2,
        output_format: OutputFormat = OutputFormat.MP3_44100_128,
        voice_settings: Optional[VoiceSettings] = None,
        language_code: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> AudioResult:
        """
        Convert text to speech using ElevenLabs TTS.
        
        Args:
            text: The text to convert to speech. Max ~5000 characters.
            voice_id: ID of the voice to use. See PreMadeVoices for options.
            model_id: TTS model to use.
            output_format: Audio output format.
            voice_settings: Optional voice settings override.
            language_code: ISO 639-1 language code (e.g., "en", "es", "ja").
            seed: For deterministic generation (0 to 4294967295).
        
        Returns:
            AudioResult containing the generated audio data.
        
        Raises:
            ElevenLabsAPIError: If the API returns an error.
        """
        client = await self._get_client()
        
        # Build request body
        body = {
            "text": text,
            "model_id": model_id.value if isinstance(model_id, TTSModel) else model_id,
        }
        
        if voice_settings:
            body["voice_settings"] = {
                "stability": voice_settings.stability,
                "similarity_boost": voice_settings.similarity_boost,
                "style": voice_settings.style,
                "use_speaker_boost": voice_settings.use_speaker_boost,
            }
        
        if language_code:
            body["language_code"] = language_code
        
        if seed is not None:
            body["seed"] = seed
        
        # Make request
        output_fmt = output_format.value if isinstance(output_format, OutputFormat) else output_format
        url = f"/text-to-speech/{voice_id}?output_format={output_fmt}"
        
        logger.info(f"Generating TTS for {len(text)} characters with voice {voice_id}")
        
        response = await client.post(url, json=body)
        
        if response.status_code != 200:
            error_text = response.text
            try:
                error_json = response.json()
                error_text = error_json.get("detail", {}).get("message", error_text)
            except Exception:
                pass
            raise ElevenLabsAPIError(response.status_code, error_text)
        
        # Extract metadata from headers
        char_cost = response.headers.get("x-character-count")
        request_id = response.headers.get("request-id")
        
        return AudioResult(
            audio_data=response.content,
            format=output_fmt,
            character_cost=int(char_cost) if char_cost else None,
            request_id=request_id,
        )
    
    # =========================================================================
    # Sound Effects Generation
    # =========================================================================
    
    async def generate_sound_effect(
        self,
        text: str,
        duration_seconds: Optional[float] = None,
        loop: bool = False,
        prompt_influence: float = 0.3,
        model_id: SFXModel = SFXModel.ELEVEN_TEXT_TO_SOUND_V2,
        output_format: OutputFormat = OutputFormat.MP3_44100_128,
    ) -> AudioResult:
        """
        Generate a sound effect from a text description.
        
        Args:
            text: Description of the sound effect to generate.
            duration_seconds: Duration of the sound (0.5 to 30 seconds).
                            If None, duration is inferred from prompt.
            loop: Whether to create a seamlessly looping sound.
            prompt_influence: How closely to follow the prompt (0.0 to 1.0).
                             Higher values = more faithful but less variable.
            model_id: Sound effects model to use.
            output_format: Audio output format.
        
        Returns:
            AudioResult containing the generated sound effect.
        
        Raises:
            ElevenLabsAPIError: If the API returns an error.
        """
        client = await self._get_client()
        
        # Build request body
        body = {
            "text": text,
            "model_id": model_id.value if isinstance(model_id, SFXModel) else model_id,
            "prompt_influence": prompt_influence,
            "loop": loop,
        }
        
        if duration_seconds is not None:
            if duration_seconds < 0.5 or duration_seconds > 30:
                raise ValueError("duration_seconds must be between 0.5 and 30")
            body["duration_seconds"] = duration_seconds
        
        # Make request
        output_fmt = output_format.value if isinstance(output_format, OutputFormat) else output_format
        url = f"/sound-generation?output_format={output_fmt}"
        
        logger.info(f"Generating sound effect: {text[:50]}...")
        
        response = await client.post(url, json=body)
        
        if response.status_code != 200:
            error_text = response.text
            try:
                error_json = response.json()
                error_text = error_json.get("detail", {}).get("message", error_text)
            except Exception:
                pass
            raise ElevenLabsAPIError(response.status_code, error_text)
        
        return AudioResult(
            audio_data=response.content,
            format=output_fmt,
        )
    
    # =========================================================================
    # Voice Management
    # =========================================================================
    
    async def get_voices(self) -> list[Voice]:
        """
        Get list of all available voices.
        
        Returns:
            List of Voice objects.
        """
        client = await self._get_client()
        
        response = await client.get("/voices")
        
        if response.status_code != 200:
            raise ElevenLabsAPIError(response.status_code, response.text)
        
        data = response.json()
        voices = []
        
        for v in data.get("voices", []):
            voices.append(Voice(
                voice_id=v["voice_id"],
                name=v["name"],
                category=v.get("category", "unknown"),
                description=v.get("description"),
                labels=v.get("labels"),
                preview_url=v.get("preview_url"),
            ))
        
        return voices
    
    async def get_voice(self, voice_id: str) -> Voice:
        """
        Get details of a specific voice.
        
        Args:
            voice_id: ID of the voice to retrieve.
        
        Returns:
            Voice object with voice details.
        """
        client = await self._get_client()
        
        response = await client.get(f"/voices/{voice_id}")
        
        if response.status_code != 200:
            raise ElevenLabsAPIError(response.status_code, response.text)
        
        v = response.json()
        
        return Voice(
            voice_id=v["voice_id"],
            name=v["name"],
            category=v.get("category", "unknown"),
            description=v.get("description"),
            labels=v.get("labels"),
            preview_url=v.get("preview_url"),
        )
    
    # =========================================================================
    # User Info
    # =========================================================================
    
    async def get_user_info(self) -> dict:
        """
        Get information about the current user/subscription.
        
        Returns:
            Dictionary containing user subscription info.
        """
        client = await self._get_client()
        
        response = await client.get("/user")
        
        if response.status_code != 200:
            raise ElevenLabsAPIError(response.status_code, response.text)
        
        return response.json()
    
    async def get_character_usage(self) -> dict:
        """
        Get character usage information.
        
        Returns:
            Dictionary with 'character_count', 'character_limit', etc.
        """
        user_info = await self.get_user_info()
        subscription = user_info.get("subscription", {})
        
        return {
            "character_count": subscription.get("character_count", 0),
            "character_limit": subscription.get("character_limit", 0),
            "characters_remaining": (
                subscription.get("character_limit", 0) - 
                subscription.get("character_count", 0)
            ),
            "tier": subscription.get("tier", "free"),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

async def generate_tts(
    text: str,
    voice_id: str = PreMadeVoices.GEORGE,
    output_path: Optional[str] = None,
) -> AudioResult:
    """
    Quick function to generate TTS audio.
    
    Args:
        text: Text to convert to speech.
        voice_id: Voice to use.
        output_path: Optional path to save the audio file.
    
    Returns:
        AudioResult with generated audio.
    """
    async with ElevenLabsClient() as client:
        result = await client.text_to_speech(text, voice_id=voice_id)
        
        if output_path:
            result.save(output_path)
        
        return result


async def generate_sfx(
    description: str,
    duration_seconds: Optional[float] = None,
    output_path: Optional[str] = None,
) -> AudioResult:
    """
    Quick function to generate sound effect.
    
    Args:
        description: Description of the sound effect.
        duration_seconds: Optional duration (0.5 to 30 seconds).
        output_path: Optional path to save the audio file.
    
    Returns:
        AudioResult with generated audio.
    """
    async with ElevenLabsClient() as client:
        result = await client.generate_sound_effect(
            text=description,
            duration_seconds=duration_seconds,
        )
        
        if output_path:
            result.save(output_path)
        
        return result
