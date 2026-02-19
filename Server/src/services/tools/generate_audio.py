"""
Generate Audio Tool for Adjoint

This tool provides AI-powered audio generation capabilities for Unity:
- Text-to-Speech (TTS): Generate character dialogue, narration
- Sound Effects (SFX): Generate game sounds from text descriptions

Provider: ElevenLabs (exclusive)

Usage via Adjoint:
    # Generate character dialogue
    await generate_audio(
        action="generate",
        audio_type="voice",
        text="Welcome to the adventure, brave hero!",
        voice_id="JBFqnCBsd6RMkjVDRZzb"
    )
    
    # Generate a sound effect
    await generate_audio(
        action="generate",
        audio_type="sfx",
        text="Sword being drawn from metal sheath",
        duration=2.0
    )
"""

import os
import asyncio
import tempfile
from typing import Annotated, Any, Literal, Optional
from pathlib import Path

from fastmcp import Context
from services.registry import adjoint_tool
from services.tools import get_unity_instance_from_context
from transport.unity_transport import send_with_unity_instance


# Import ElevenLabs client
try:
    from integrations.elevenlabs_client import (
        ElevenLabsClient,
        ElevenLabsError,
        ElevenLabsAPIError,
        AudioResult,
        TTSModel,
        SFXModel,
        OutputFormat,
        VoiceSettings,
        PreMadeVoices,
    )
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False


# =============================================================================
# Type Definitions
# =============================================================================

AudioType = Literal["voice", "sfx", "ambient"]
Action = Literal["generate", "list_voices", "check_usage"]
TTSModelType = Literal["multilingual_v2", "flash_v2_5", "turbo_v2_5", "v3"]


# =============================================================================
# Voice Presets for Game Development
# =============================================================================

VOICE_PRESETS = {
    # Male voices
    "hero_male": PreMadeVoices.JOSH,          # Deep, heroic
    "narrator": PreMadeVoices.GEORGE,         # British, authoritative
    "villain": PreMadeVoices.ARNOLD,          # Crisp, menacing
    "merchant": PreMadeVoices.CHARLIE,        # Casual, friendly
    "warrior": PreMadeVoices.ADAM,            # Deep, strong
    
    # Female voices
    "hero_female": PreMadeVoices.BELLA,       # Soft, determined
    "narrator_female": PreMadeVoices.EMILY,   # British, professional
    "mentor": PreMadeVoices.RACHEL,           # Calm, wise
    "companion": PreMadeVoices.ELLI,          # Emotional, expressive
}

# Map model names to enums
MODEL_MAP = {
    "multilingual_v2": TTSModel.ELEVEN_MULTILINGUAL_V2,
    "flash_v2_5": TTSModel.ELEVEN_FLASH_V2_5,
    "turbo_v2_5": TTSModel.ELEVEN_TURBO_V2_5,
    "v3": "eleven_v3",  # V3 is in alpha, use string
}

# Recommended settings by use case
VOICE_SETTINGS_PRESETS = {
    "dialogue": VoiceSettings(
        stability=0.4,
        similarity_boost=0.75,
        style=0.15,
        use_speaker_boost=True
    ),
    "narration": VoiceSettings(
        stability=0.7,
        similarity_boost=0.8,
        style=0.0,
        use_speaker_boost=True
    ),
    "emotional": VoiceSettings(
        stability=0.25,
        similarity_boost=0.7,
        style=0.3,
        use_speaker_boost=True
    ),
    "ui": VoiceSettings(
        stability=0.85,
        similarity_boost=0.5,
        style=0.0,
        use_speaker_boost=False
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_voice_id(voice: str) -> str:
    """
    Resolve voice parameter to a voice ID.
    Accepts: preset name, voice ID, or voice name.
    """
    # Check if it's a preset
    if voice.lower() in VOICE_PRESETS:
        return VOICE_PRESETS[voice.lower()]
    
    # Assume it's a voice ID if it looks like one (contains alphanumeric)
    if len(voice) > 10 and voice.isalnum():
        return voice
    
    # Default to George (narrator) if unknown
    return PreMadeVoices.GEORGE


def get_output_path(
    audio_type: str,
    filename: str,
    target_folder: Optional[str] = None
) -> str:
    """Generate the Unity asset path for the audio file."""
    if target_folder:
        folder = target_folder.rstrip("/")
    else:
        folder = f"Assets/Generated/Audio/{audio_type.capitalize()}"
    
    # Ensure .mp3 extension
    if not filename.endswith(".mp3"):
        filename = f"{filename}.mp3"
    
    return f"{folder}/{filename}"


async def save_audio_to_temp(audio_result: AudioResult, prefix: str = "audio") -> str:
    """Save audio bytes to a temporary file and return the path."""
    suffix = f".{audio_result.format.split('_')[0]}"  # Extract format (mp3, pcm, etc.)
    
    with tempfile.NamedTemporaryFile(
        delete=False,
        prefix=f"{prefix}_",
        suffix=suffix
    ) as f:
        f.write(audio_result.audio_data)
        return f.name


# =============================================================================
# Main Tool Implementation
# =============================================================================

@adjoint_tool(
    name="generate_audio",
    description="""Generate audio for Unity games using AI.

Supports:
- Voice/TTS: Character dialogue, narration, NPC barks
- Sound Effects: Game SFX from text descriptions
- Ambient: Looping environmental sounds

Actions:
- generate: Create new audio
- list_voices: Show available voice presets and IDs
- check_usage: Show remaining credits

Examples:
- Generate hero dialogue: audio_type="voice", text="For glory!", voice="hero_male"
- Generate sword SFX: audio_type="sfx", text="Sharp sword slash", duration=1.5
- Generate ambient: audio_type="ambient", text="Forest with birds", duration=30, loop=True
"""
)
async def generate_audio(
    ctx: Context,
    action: Annotated[
        Action,
        "Action to perform: 'generate', 'list_voices', or 'check_usage'"
    ] = "generate",
    
    # Generation parameters
    audio_type: Annotated[
        AudioType,
        "Type of audio: 'voice' (TTS), 'sfx' (sound effects), or 'ambient' (looping)"
    ] = "voice",
    text: Annotated[
        Optional[str],
        "Text to convert to speech OR description of sound effect"
    ] = None,
    
    # Voice-specific parameters
    voice: Annotated[
        Optional[str],
        "Voice preset (hero_male, narrator, etc.), voice ID, or voice name"
    ] = "narrator",
    model: Annotated[
        Optional[TTSModelType],
        "TTS model: 'multilingual_v2' (quality), 'flash_v2_5' (fast), 'turbo_v2_5', 'v3' (expressive)"
    ] = "multilingual_v2",
    style: Annotated[
        Optional[str],
        "Voice style preset: 'dialogue', 'narration', 'emotional', 'ui'"
    ] = "dialogue",
    
    # SFX-specific parameters
    duration: Annotated[
        Optional[float],
        "Duration in seconds for SFX (0.5-30). None = auto-detect from prompt."
    ] = None,
    loop: Annotated[
        bool,
        "Create seamlessly looping audio (for ambient sounds)"
    ] = False,
    prompt_influence: Annotated[
        float,
        "How closely to follow the prompt (0-1). Higher = more literal."
    ] = 0.3,
    
    # Output parameters
    filename: Annotated[
        Optional[str],
        "Output filename (without extension). Auto-generated if not provided."
    ] = None,
    target_folder: Annotated[
        Optional[str],
        "Unity folder path. Default: Assets/Generated/Audio/{type}/"
    ] = None,
    import_to_unity: Annotated[
        bool,
        "Whether to import the audio into Unity"
    ] = True,
) -> dict[str, Any]:
    """
    Generate audio using ElevenLabs AI.
    
    Returns dict with:
    - success: bool
    - audio_type: Type of audio generated
    - file_path: Local temp file path
    - unity_path: Unity asset path (if imported)
    - duration: Estimated duration
    - credits_used: Characters/credits consumed
    - error: Error message if failed
    """
    
    # =========================================================================
    # Action: List Voices
    # =========================================================================
    if action == "list_voices":
        await ctx.info("📋 Available voice presets and IDs")
        
        voices_info = {
            "presets": {
                name: voice_id for name, voice_id in VOICE_PRESETS.items()
            },
            "style_presets": list(VOICE_SETTINGS_PRESETS.keys()),
            "models": list(MODEL_MAP.keys()),
        }
        
        # Try to get user's voices from ElevenLabs
        if ELEVENLABS_AVAILABLE:
            try:
                async with ElevenLabsClient() as client:
                    user_voices = await client.get_voices()
                    voices_info["user_voices"] = [
                        {"id": v.voice_id, "name": v.name, "category": v.category}
                        for v in user_voices[:10]  # Limit to 10
                    ]
            except Exception:
                pass
        
        return {
            "success": True,
            "voices": voices_info
        }
    
    # =========================================================================
    # Action: Check Usage
    # =========================================================================
    if action == "check_usage":
        if not ELEVENLABS_AVAILABLE:
            return {
                "success": False,
                "error": "ElevenLabs client not available"
            }
        
        await ctx.info("📊 Checking ElevenLabs usage...")
        
        try:
            async with ElevenLabsClient() as client:
                usage = await client.get_character_usage()
                return {
                    "success": True,
                    "usage": {
                        "characters_used": usage["character_count"],
                        "characters_limit": usage["character_limit"],
                        "characters_remaining": usage["characters_remaining"],
                        "tier": usage["tier"],
                    }
                }
        except ElevenLabsError as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # =========================================================================
    # Action: Generate Audio
    # =========================================================================
    if action != "generate":
        return {
            "success": False,
            "error": f"Unknown action: {action}"
        }
    
    # Validate text input
    if not text or not text.strip():
        return {
            "success": False,
            "error": "Text parameter is required for generation"
        }
    
    if not ELEVENLABS_AVAILABLE:
        return {
            "success": False,
            "error": "ElevenLabs client not available. Check ELEVENLABS_API_KEY."
        }
    
    # Generate filename if not provided
    if not filename:
        # Create filename from first few words of text
        words = text.strip().split()[:4]
        filename = "_".join(words).lower()
        # Clean up filename
        filename = "".join(c if c.isalnum() or c == "_" else "" for c in filename)
        filename = filename[:50]  # Limit length
    
    try:
        async with ElevenLabsClient() as client:
            
            # -----------------------------------------------------------------
            # Voice/TTS Generation
            # -----------------------------------------------------------------
            if audio_type == "voice":
                voice_id = get_voice_id(voice or "narrator")
                voice_settings = VOICE_SETTINGS_PRESETS.get(
                    style or "dialogue",
                    VOICE_SETTINGS_PRESETS["dialogue"]
                )
                
                # Get model
                model_id = MODEL_MAP.get(model, TTSModel.ELEVEN_MULTILINGUAL_V2)
                
                await ctx.info(f"🎙️ Generating voice: {text[:50]}...")
                
                result = await client.text_to_speech(
                    text=text,
                    voice_id=voice_id,
                    model_id=model_id,
                    voice_settings=voice_settings,
                    output_format=OutputFormat.MP3_44100_128,
                )
                
                # Estimate duration (~150 words per minute, ~5 chars per word)
                estimated_duration = len(text) / (150 * 5 / 60)
                credits_used = result.character_cost or len(text)
                
            # -----------------------------------------------------------------
            # Sound Effects Generation
            # -----------------------------------------------------------------
            elif audio_type in ("sfx", "ambient"):
                is_ambient = audio_type == "ambient"
                
                # Validate duration
                if duration is not None:
                    if duration < 0.5 or duration > 30:
                        return {
                            "success": False,
                            "error": "Duration must be between 0.5 and 30 seconds"
                        }
                
                # For ambient, default to 30 seconds with looping
                if is_ambient:
                    duration = duration or 30.0
                    loop = True
                
                await ctx.info(f"🔊 Generating {audio_type}: {text[:50]}...")
                
                result = await client.generate_sound_effect(
                    text=text,
                    duration_seconds=duration,
                    loop=loop,
                    prompt_influence=prompt_influence,
                    output_format=OutputFormat.MP3_44100_128,
                )
                
                # Calculate credits (40 per second for SFX)
                estimated_duration = duration or 5.0
                credits_used = int(estimated_duration * 40)
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown audio_type: {audio_type}"
                }
            
            # -----------------------------------------------------------------
            # Save to temp file
            # -----------------------------------------------------------------
            temp_path = await save_audio_to_temp(result, prefix=audio_type)
            await ctx.info(f"💾 Saved to: {temp_path}")
            
            # -----------------------------------------------------------------
            # Import to Unity
            # -----------------------------------------------------------------
            unity_path = None
            if import_to_unity:
                unity_path = get_output_path(audio_type, filename, target_folder)
                
                # Get Unity instance
                unity_instance = await get_unity_instance_from_context(ctx)
                if unity_instance:
                    try:
                        await ctx.info(f"📦 Importing to Unity: {unity_path}")
                        
                        # Send import command to Unity
                        import_result = await send_with_unity_instance(
                            unity_instance,
                            {
                                "type": "import_audio",
                                "source_path": temp_path,
                                "target_path": unity_path,
                                "audio_type": audio_type,
                                "loop": loop,
                            }
                        )
                        
                        if import_result.get("success"):
                            await ctx.info(f"✅ Imported to Unity: {unity_path}")
                        else:
                            await ctx.warning(
                                f"⚠️ Unity import pending. File saved at: {temp_path}"
                            )
                    except Exception as e:
                        await ctx.warning(f"⚠️ Could not import to Unity: {e}")
                else:
                    await ctx.warning("⚠️ No Unity connection. File saved locally.")
            
            # -----------------------------------------------------------------
            # Return result
            # -----------------------------------------------------------------
            return {
                "success": True,
                "audio_type": audio_type,
                "text": text[:100] + ("..." if len(text) > 100 else ""),
                "file_path": temp_path,
                "unity_path": unity_path,
                "duration_seconds": round(estimated_duration, 2),
                "credits_used": credits_used,
                "loop": loop,
                "voice": voice if audio_type == "voice" else None,
                "model": model if audio_type == "voice" else None,
            }
            
    except ElevenLabsAPIError as e:
        await ctx.error(f"❌ ElevenLabs API error: {e.message}")
        return {
            "success": False,
            "error": f"API error ({e.status_code}): {e.message}"
        }
    except ElevenLabsError as e:
        await ctx.error(f"❌ ElevenLabs error: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        await ctx.error(f"❌ Unexpected error: {e}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


# =============================================================================
# Convenience Tools
# =============================================================================

@adjoint_tool(
    name="generate_dialogue",
    description="Quick tool to generate character dialogue. Shortcut for generate_audio with voice type."
)
async def generate_dialogue(
    ctx: Context,
    text: Annotated[str, "The dialogue text to speak"],
    character: Annotated[
        str,
        "Character type: 'hero_male', 'hero_female', 'narrator', 'villain', 'merchant', 'warrior', 'mentor', 'companion'"
    ] = "hero_male",
    emotion: Annotated[
        str,
        "Emotional style: 'dialogue' (normal), 'emotional' (expressive), 'narration' (neutral)"
    ] = "dialogue",
    filename: Annotated[Optional[str], "Output filename"] = None,
) -> dict[str, Any]:
    """Generate character dialogue with preset voices."""
    return await generate_audio(
        ctx=ctx,
        action="generate",
        audio_type="voice",
        text=text,
        voice=character,
        style=emotion,
        filename=filename,
        import_to_unity=True,
    )


@adjoint_tool(
    name="generate_sfx",
    description="Quick tool to generate sound effects. Shortcut for generate_audio with sfx type."
)
async def generate_sfx(
    ctx: Context,
    description: Annotated[str, "Description of the sound effect"],
    duration: Annotated[
        Optional[float],
        "Duration in seconds (0.5-30). None = auto-detect."
    ] = None,
    filename: Annotated[Optional[str], "Output filename"] = None,
) -> dict[str, Any]:
    """Generate a sound effect from text description."""
    return await generate_audio(
        ctx=ctx,
        action="generate",
        audio_type="sfx",
        text=description,
        duration=duration,
        filename=filename,
        import_to_unity=True,
    )


@adjoint_tool(
    name="generate_ambient",
    description="Quick tool to generate ambient/environmental audio. Creates looping sounds."
)
async def generate_ambient(
    ctx: Context,
    description: Annotated[str, "Description of the ambient sound/atmosphere"],
    duration: Annotated[float, "Duration in seconds (max 30)"] = 30.0,
    filename: Annotated[Optional[str], "Output filename"] = None,
) -> dict[str, Any]:
    """Generate looping ambient audio."""
    return await generate_audio(
        ctx=ctx,
        action="generate",
        audio_type="ambient",
        text=description,
        duration=duration,
        loop=True,
        filename=filename,
        import_to_unity=True,
    )
