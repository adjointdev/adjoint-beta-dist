"""
Adjoint AI Integrations Module

This module provides async clients for various AI generation services:
- Meshy: 3D model generation with auto-rigging and animation
- ElevenLabs: Text-to-speech and sound effects generation

Asset Type Routing:
- Essential assets (player, enemy, character, weapon) → LATEST (highest quality)
- Environment assets (tree, rock, building, prop) → Meshy-5 (good quality, faster)

Animation System:
- Basic animations (walking, running) are FREE with rigging
- Additional animations (jump, attack, dance) use the Animation Library API

Usage:
    from integrations import MeshyClient, ElevenLabsClient, classify_asset, get_model_for_asset
    
    # For 3D models with auto-routing
    async with MeshyClient() as client:
        # Automatically selects LATEST (Meshy-6) for essential, Meshy-5 for environment
        result = await client.generate_3d_model("a brave knight warrior")
        
        # Generate animated character with rigging
        character = await client.generate_animated_character(
            "a medieval knight",
            animations=["walk", "run", "sword slash", "death"]
        )
    
    # For audio - use ElevenLabs
    async with ElevenLabsClient() as client:
        result = await client.text_to_speech("Welcome, brave hero!")
        sfx = await client.generate_sound_effect("sword slash")
"""

from .meshy_client import (
    MeshyClient,
    MeshyTask,
    MeshyError,
    MeshyAPIError,
    MeshyTaskError,
    MeshyAIModel,
    ArtStyle,
    Topology,
    SymmetryMode,
    PoseMode,
    TaskStatus as MeshyTaskStatus,
    TextureUrls,
    ModelUrls,
    AssetType,
    AnimationAction,
    BasicAnimations,
    AnimationLibrary,
    classify_asset,
    get_model_for_asset,
    is_basic_animation,
    get_animation_action_id,
)

from .elevenlabs_client import (
    ElevenLabsClient,
    ElevenLabsError,
    ElevenLabsAPIError,
    ElevenLabsConfigError,
    AudioResult,
    Voice,
    VoiceSettings,
    OutputFormat,
    TTSModel,
    SFXModel,
    PreMadeVoices,
    generate_tts,
    generate_sfx,
)

__all__ = [
    # Meshy (PRIMARY - all 3D generation with routing)
    "MeshyClient",
    "MeshyTask",
    "MeshyError",
    "MeshyAPIError",
    "MeshyTaskError",
    "MeshyAIModel",
    "ArtStyle",
    "Topology",
    "SymmetryMode",
    "PoseMode",
    "MeshyTaskStatus",
    "TextureUrls",
    "ModelUrls",
    "AssetType",
    "AnimationAction",
    "BasicAnimations",
    "AnimationLibrary",
    "classify_asset",
    "get_model_for_asset",
    "is_basic_animation",
    "get_animation_action_id",
    
    # ElevenLabs (Audio - TTS and SFX)
    "ElevenLabsClient",
    "ElevenLabsError",
    "ElevenLabsAPIError",
    "ElevenLabsConfigError",
    "AudioResult",
    "Voice",
    "VoiceSettings",
    "OutputFormat",
    "TTSModel",
    "SFXModel",
    "PreMadeVoices",
    "generate_tts",
    "generate_sfx",
]
