"""
Meshy.ai Client for 3D Model Generation API

This client provides async integration with Meshy.ai's API for:
- Text-to-3D generation (preview + refine workflow)
- Image-to-3D generation (stub - not yet implemented)
- Multi-image-to-3D generation
- Rigging (auto-rig humanoid models)
- Animation (apply animations to rigged models)
- Retexture (restyle existing models)
- Remesh (optimize mesh topology)
- Text-to-Image (stub - not yet implemented)

Routing Strategy:
- Essential assets (player, enemy, character, weapon) → LATEST/Meshy-6 (highest quality)
- Environment assets (tree, rock, building, prop) → Meshy-5 (good quality, faster)

API Documentation: See development_docs/MeshyAIDocumentation.txt
"""

import os
import asyncio
import time
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

import httpx


# =============================================================================
# Enums
# =============================================================================

class MeshyAIModel(str, Enum):
    """Available AI models for generation."""
    MESHY_5 = "meshy-5"
    MESHY_6 = "meshy-6"
    LATEST = "latest"  # Currently points to Meshy-6 (as of 2024)


class ArtStyle(str, Enum):
    """Art style for Text-to-3D."""
    REALISTIC = "realistic"
    SCULPTURE = "sculpture"


class Topology(str, Enum):
    """Mesh topology type."""
    TRIANGLE = "triangle"
    QUAD = "quad"


class SymmetryMode(str, Enum):
    """Symmetry enforcement mode."""
    OFF = "off"
    AUTO = "auto"
    ON = "on"


class PoseMode(str, Enum):
    """Pose mode for character rigging preparation."""
    NONE = ""
    A_POSE = "a-pose"
    T_POSE = "t-pose"


class TaskStatus(str, Enum):
    """Task processing status."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class AssetType(str, Enum):
    """Asset classification for routing decisions."""
    ESSENTIAL = "essential"      # Player, enemy, character, weapon → LATEST/Meshy-6 (highest quality)
    ENVIRONMENT = "environment"  # Tree, rock, building, prop → Meshy-5


def _enum_value(val) -> str:
    """Extract string value from enum or return string as-is."""
    return val.value if hasattr(val, 'value') else str(val)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TextureUrls:
    """PBR texture map URLs."""
    base_color: Optional[str] = None
    metallic: Optional[str] = None
    normal: Optional[str] = None
    roughness: Optional[str] = None


@dataclass
class ModelUrls:
    """3D model download URLs."""
    glb: Optional[str] = None
    fbx: Optional[str] = None
    obj: Optional[str] = None
    mtl: Optional[str] = None
    usdz: Optional[str] = None
    pre_remeshed_glb: Optional[str] = None


@dataclass
class BasicAnimations:
    """
    Basic animations included with rigging (walking and running).
    These are provided automatically with rigging - no API call needed.
    """
    walking_glb_url: Optional[str] = None
    walking_fbx_url: Optional[str] = None
    walking_armature_glb_url: Optional[str] = None
    running_glb_url: Optional[str] = None
    running_fbx_url: Optional[str] = None
    running_armature_glb_url: Optional[str] = None
    
    @property
    def has_walking(self) -> bool:
        return bool(self.walking_glb_url or self.walking_fbx_url)
    
    @property
    def has_running(self) -> bool:
        return bool(self.running_glb_url or self.running_fbx_url)


# Common animation action IDs from Meshy's animation library
# Use these with animate_model() for animations beyond basic walking/running
class AnimationLibrary:
    """
    Meshy Animation Library action IDs.
    
    Basic animations (walking, running) are included FREE with rigging.
    Use these IDs with animate_model() for additional animations.
    """
    # Locomotion
    IDLE = 92
    WALK = 93
    RUN = 94
    SPRINT = 95
    
    # Combat
    PUNCH = 100
    KICK = 101
    SWORD_SLASH = 102
    SWORD_STAB = 103
    BOW_SHOOT = 104
    
    # Actions
    JUMP = 110
    CROUCH = 111
    ROLL = 112
    CLIMB = 113
    
    # Emotes
    WAVE = 120
    DANCE = 121
    CHEER = 122
    TAUNT = 123
    
    # Deaths/Reactions
    HIT_REACTION = 130
    DEATH_FORWARD = 131
    DEATH_BACKWARD = 132


# Animations that are included FREE with rigging (no API call needed)
BASIC_ANIMATION_NAMES = {"walk", "walking", "run", "running"}


def is_basic_animation(animation_name: str) -> bool:
    """
    Check if an animation is included free with rigging.
    
    Walking and Running are provided automatically - no extra API call needed.
    """
    return animation_name.lower().strip() in BASIC_ANIMATION_NAMES


def get_animation_action_id(animation_name: str) -> Optional[int]:
    """
    Get the Meshy animation library action ID for an animation name.
    
    Returns None if the animation name isn't recognized.
    """
    name_lower = animation_name.lower().strip()
    
    # Map common names to action IDs
    animation_map = {
        "idle": AnimationLibrary.IDLE,
        "stand": AnimationLibrary.IDLE,
        "walk": AnimationLibrary.WALK,
        "walking": AnimationLibrary.WALK,
        "run": AnimationLibrary.RUN,
        "running": AnimationLibrary.RUN,
        "sprint": AnimationLibrary.SPRINT,
        "sprinting": AnimationLibrary.SPRINT,
        "punch": AnimationLibrary.PUNCH,
        "kick": AnimationLibrary.KICK,
        "slash": AnimationLibrary.SWORD_SLASH,
        "sword": AnimationLibrary.SWORD_SLASH,
        "stab": AnimationLibrary.SWORD_STAB,
        "shoot": AnimationLibrary.BOW_SHOOT,
        "bow": AnimationLibrary.BOW_SHOOT,
        "jump": AnimationLibrary.JUMP,
        "crouch": AnimationLibrary.CROUCH,
        "roll": AnimationLibrary.ROLL,
        "dodge": AnimationLibrary.ROLL,
        "climb": AnimationLibrary.CLIMB,
        "wave": AnimationLibrary.WAVE,
        "dance": AnimationLibrary.DANCE,
        "dancing": AnimationLibrary.DANCE,
        "cheer": AnimationLibrary.CHEER,
        "taunt": AnimationLibrary.TAUNT,
        "hit": AnimationLibrary.HIT_REACTION,
        "hurt": AnimationLibrary.HIT_REACTION,
        "death": AnimationLibrary.DEATH_FORWARD,
        "die": AnimationLibrary.DEATH_FORWARD,
    }
    
    return animation_map.get(name_lower)


@dataclass
class MeshyTask:
    """Represents a Meshy generation task."""
    id: str
    task_type: str  # "text-to-3d", "image-to-3d", "rigging", "animation", etc.
    status: TaskStatus
    progress: int = 0
    model_urls: Optional[ModelUrls] = None
    texture_urls: List[TextureUrls] = field(default_factory=list)
    thumbnail_url: Optional[str] = None
    prompt: Optional[str] = None
    art_style: Optional[str] = None
    started_at: Optional[int] = None
    created_at: Optional[int] = None
    finished_at: Optional[int] = None
    preceding_tasks: int = 0
    error_message: Optional[str] = None
    
    # Rigging-specific fields
    rig_id: Optional[str] = None
    basic_animations: Optional[BasicAnimations] = None  # Walking/running included with rigging
    
    # Animation-specific fields
    animation_glb_url: Optional[str] = None
    animation_fbx_url: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        return self.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED)
    
    @property
    def is_success(self) -> bool:
        return self.status == TaskStatus.SUCCEEDED
    
    @property
    def glb_url(self) -> Optional[str]:
        """Get GLB download URL if available."""
        return self.model_urls.glb if self.model_urls else None
    
    @property
    def fbx_url(self) -> Optional[str]:
        """Get FBX download URL if available."""
        return self.model_urls.fbx if self.model_urls else None
    
    @property
    def has_basic_animations(self) -> bool:
        """Check if basic animations (walking/running) are available."""
        return self.basic_animations is not None and (
            self.basic_animations.has_walking or self.basic_animations.has_running
        )


@dataclass
class AnimationAction:
    """Represents an animation from the Meshy animation library."""
    action_id: int
    name: str
    category: str
    duration_seconds: float = 0.0


# =============================================================================
# Asset Type Classification
# =============================================================================

# Keywords for classifying assets
ESSENTIAL_KEYWORDS = [
    # Characters
    "player", "character", "hero", "protagonist", "npc", "human", "humanoid", "person",
    "warrior", "knight", "mage", "wizard", "archer", "soldier", "guard",
    # Enemies
    "enemy", "monster", "creature", "boss", "villain", "demon", "dragon",
    "zombie", "skeleton", "goblin", "orc", "troll", "beast",
    # Weapons & Equipment
    "weapon", "sword", "axe", "bow", "staff", "wand", "shield", "armor",
    "gun", "rifle", "pistol", "knife", "dagger",
    # Vehicles (player-controlled)
    "vehicle", "car", "spaceship", "mech", "robot",
]

ENVIRONMENT_KEYWORDS = [
    # Nature
    "tree", "rock", "stone", "boulder", "bush", "plant", "flower", "grass",
    "mountain", "hill", "cliff", "cave", "forest", "wood",
    # Buildings
    "building", "house", "castle", "tower", "wall", "fence", "gate", "door",
    "bridge", "road", "path", "floor", "ceiling", "window",
    # Props
    "prop", "crate", "barrel", "box", "chest", "table", "chair", "bench",
    "lamp", "torch", "sign", "flag", "statue", "fountain",
    # Environment
    "environment", "terrain", "landscape", "background", "scenery",
    "decoration", "decor", "furniture",
]


def classify_asset(prompt: str) -> AssetType:
    """
    Classify an asset based on its prompt to determine which model to use.
    
    Args:
        prompt: The text description of the 3D model.
        
    Returns:
        AssetType.ESSENTIAL for important assets (use LATEST/Meshy-6 - highest quality)
        AssetType.ENVIRONMENT for background assets (use Meshy-5)
    """
    prompt_lower = prompt.lower()
    
    # Check for essential keywords first (higher priority)
    for keyword in ESSENTIAL_KEYWORDS:
        if keyword in prompt_lower:
            return AssetType.ESSENTIAL
    
    # Check for environment keywords
    for keyword in ENVIRONMENT_KEYWORDS:
        if keyword in prompt_lower:
            return AssetType.ENVIRONMENT
    
    # Default to essential (safer - use higher quality)
    return AssetType.ESSENTIAL


def get_model_for_asset(prompt: str, force_model: Optional[MeshyAIModel] = None) -> MeshyAIModel:
    """
    Determine which Meshy model to use based on asset classification.

    NOTE: Routing is currently DISABLED - all assets use Meshy-5 for cost efficiency.
    Previously: Essential assets → Meshy-6, Environment assets → Meshy-5

    Args:
        prompt: The text description of the 3D model.
        force_model: Override automatic selection with specific model.

    Returns:
        MeshyAIModel to use for generation.
    """
    if force_model:
        return force_model

    # Routing disabled - always use Meshy-5 for cost efficiency
    return MeshyAIModel.MESHY_5


# =============================================================================
# Main Client
# =============================================================================

class MeshyClient:
    """
    Async client for Meshy.ai API.
    
    Features:
    - Text-to-3D with automatic model routing (LATEST/Meshy-6 for essential, Meshy-5 for environment)
    - Image-to-3D (single image and multi-image)
    - Rigging for humanoid models
    - Animation application
    - Retexturing existing models
    - Webhook support for progress updates
    
    Usage:
        async with MeshyClient() as client:
            # Text-to-3D with automatic routing
            result = await client.generate_3d_model("a warrior character")  # Uses LATEST (Meshy-6)
            result = await client.generate_3d_model("a wooden barrel")      # Uses Meshy-5
            
            # Force specific model
            result = await client.generate_3d_model("a tree", ai_model=MeshyAIModel.MESHY_5)
            
            # Rig a character for animation
            rig_result = await client.rig_model(model_url=result.glb_url)
            
            # Apply animation
            anim_result = await client.animate_model(rig_result.id, action_id=92)
    """
    
    BASE_URL = "https://api.meshy.ai"
    
    # API versions
    TEXT_TO_3D_VERSION = "v2"
    IMAGE_TO_3D_VERSION = "v1"
    MULTI_IMAGE_VERSION = "v1"
    RIGGING_VERSION = "v1"
    ANIMATION_VERSION = "v1"
    RETEXTURE_VERSION = "v1"
    REMESH_VERSION = "v1"
    TEXT_TO_IMAGE_VERSION = "v1"
    
    # Defaults - polycount does NOT affect credit cost, only model choice does
    DEFAULT_POLYCOUNT = 100_000  # High quality default (max: 300k, Meshy default: 30k)
    MAX_POLYCOUNT = 300_000
    MIN_POLYCOUNT = 100
    
    # Timeouts
    DEFAULT_TIMEOUT = 600.0  # 10 minutes (refine can take a while)
    POLL_INTERVAL = 5.0      # 5 seconds
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ):
        """
        Initialize Meshy.ai client.
        
        Args:
            api_key: Meshy API key. Defaults to env MESHY_API_KEY.
            webhook_url: URL for webhook callbacks. Defaults to env WEBHOOK_URL.
            webhook_secret: Secret for webhook validation. Defaults to env WEBHOOK_SECRET.
        """
        self.api_key = api_key or os.getenv("MESHY_API_KEY")
        self.webhook_url = webhook_url or os.getenv("WEBHOOK_URL")
        self.webhook_secret = webhook_secret or os.getenv("WEBHOOK_SECRET")
        
        if not self.api_key:
            raise ValueError(
                "Meshy API key required. Set MESHY_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        self._client: Optional[httpx.AsyncClient] = None
        
        # Callback for webhook events (set by external handler)
        self._webhook_callback: Optional[Callable[[dict], None]] = None
    
    async def __aenter__(self) -> "MeshyClient":
        self._client = httpx.AsyncClient(timeout=60.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
    
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client
    
    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _parse_model_urls(self, data: Dict[str, Any]) -> Optional[ModelUrls]:
        """Parse model_urls from API response."""
        urls = data.get("model_urls")
        if not urls:
            return None
        return ModelUrls(
            glb=urls.get("glb"),
            fbx=urls.get("fbx"),
            obj=urls.get("obj"),
            mtl=urls.get("mtl"),
            usdz=urls.get("usdz"),
            pre_remeshed_glb=urls.get("pre_remeshed_glb"),
        )
    
    def _parse_texture_urls(self, data: Dict[str, Any]) -> List[TextureUrls]:
        """Parse texture_urls array from API response."""
        textures = data.get("texture_urls") or []
        result = []
        for t in textures:
            if t:
                result.append(TextureUrls(
                    base_color=t.get("base_color"),
                    metallic=t.get("metallic"),
                    normal=t.get("normal"),
                    roughness=t.get("roughness"),
                ))
        return result
    
    def _parse_basic_animations(self, data: Dict[str, Any]) -> Optional[BasicAnimations]:
        """Parse basic_animations from rigging task response."""
        result = data.get("result", {})
        if not result:
            return None
        
        basic_anims = result.get("basic_animations", {})
        if not basic_anims:
            return None
        
        return BasicAnimations(
            walking_glb_url=basic_anims.get("walking_glb_url"),
            walking_fbx_url=basic_anims.get("walking_fbx_url"),
            walking_armature_glb_url=basic_anims.get("walking_armature_glb_url"),
            running_glb_url=basic_anims.get("running_glb_url"),
            running_fbx_url=basic_anims.get("running_fbx_url"),
            running_armature_glb_url=basic_anims.get("running_armature_glb_url"),
        )
    
    def _parse_rigging_model_urls(self, data: Dict[str, Any]) -> Optional[ModelUrls]:
        """Parse model URLs from rigging task response (different structure)."""
        result = data.get("result", {})
        if not result:
            return None
        
        return ModelUrls(
            glb=result.get("rigged_character_glb_url"),
            fbx=result.get("rigged_character_fbx_url"),
        )
    
    def _parse_animation_urls(self, data: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        """Parse animation URLs from animation task response."""
        result = data.get("result", {})
        if not result:
            return None, None
        
        return result.get("animation_glb_url"), result.get("animation_fbx_url")
    
    def _parse_task(self, data: Dict[str, Any]) -> MeshyTask:
        """Parse task response into MeshyTask dataclass."""
        error_msg = None
        task_error = data.get("task_error", {})
        if task_error:
            error_msg = task_error.get("message")
        
        task_type = data.get("type", "")
        
        # Handle different response structures based on task type
        if task_type == "rig":
            model_urls = self._parse_rigging_model_urls(data)
            basic_animations = self._parse_basic_animations(data)
            animation_glb, animation_fbx = None, None
        elif task_type == "animate":
            model_urls = self._parse_model_urls(data)
            basic_animations = None
            animation_glb, animation_fbx = self._parse_animation_urls(data)
        else:
            model_urls = self._parse_model_urls(data)
            basic_animations = None
            animation_glb, animation_fbx = None, None
        
        return MeshyTask(
            id=data.get("id", ""),
            task_type=task_type,
            status=TaskStatus(data.get("status", "PENDING")),
            progress=data.get("progress", 0),
            model_urls=model_urls,
            texture_urls=self._parse_texture_urls(data),
            thumbnail_url=data.get("thumbnail_url"),
            prompt=data.get("prompt"),
            art_style=data.get("art_style"),
            started_at=data.get("started_at"),
            created_at=data.get("created_at"),
            finished_at=data.get("finished_at"),
            preceding_tasks=data.get("preceding_tasks", 0),
            error_message=error_msg,
            rig_id=data.get("rig_id"),
            basic_animations=basic_animations,
            animation_glb_url=animation_glb,
            animation_fbx_url=animation_fbx,
        )
    
    # =========================================================================
    # High-Level API (with automatic routing)
    # =========================================================================
    
    async def generate_3d_model(
        self,
        prompt: str,
        *,
        ai_model: Optional[MeshyAIModel] = None,
        art_style: ArtStyle = ArtStyle.REALISTIC,
        enable_pbr: bool = True,
        pose_mode: PoseMode = PoseMode.NONE,
        target_polycount: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MeshyTask:
        """
        Generate a 3D model from text with automatic model routing.

        If target_polycount is None, smart defaults are applied:
        - Low-poly/toon/stylized prompts: 40,000
        - Humanoid characters: 150,000
        - Environment assets: 80,000
        - Other props: 50,000

        This is the main entry point for text-to-3D generation. It automatically:
        1. Classifies the asset (essential vs environment)
        2. Selects appropriate Meshy model (LATEST/Meshy-6 for essential, Meshy-5 for environment)
        3. Runs preview → refine workflow
        4. Returns the final textured model
        
        Args:
            prompt: Description of the 3D model (max 600 chars).
            ai_model: Override automatic model selection.
            art_style: Realistic or sculpture style.
            enable_pbr: Generate PBR texture maps.
            pose_mode: A-pose/T-pose for characters (for later rigging).
            target_polycount: Target polygon count (100 - 300,000).
            timeout: Maximum total time for both stages.
            progress_callback: Optional callback for progress updates (0-1, message).
            
        Returns:
            Completed MeshyTask with model URLs.
            
        Example:
            # Automatic routing
            character = await client.generate_3d_model("a knight in armor")  # Uses LATEST (Meshy-6)
            prop = await client.generate_3d_model("a wooden barrel")         # Uses Meshy-5
            
            # Force specific model
            tree = await client.generate_3d_model("an oak tree", ai_model=MeshyAIModel.MESHY_6)
        """
        # Auto-select model based on prompt classification
        selected_model = get_model_for_asset(prompt, ai_model)
        asset_type = classify_asset(prompt)

        # Smart polycount defaults based on asset type
        if target_polycount is None:
            _HUMANOID_KEYWORDS = [
                "character", "human", "humanoid", "person", "warrior", "knight",
                "mage", "wizard", "archer", "soldier", "guard", "npc",
                "hero", "protagonist", "villain", "enemy", "monster", "creature",
                "zombie", "skeleton", "demon", "dragon", "player",
            ]
            prompt_lower = prompt.lower()
            if any(kw in prompt_lower for kw in ("low poly", "lowpoly", "low-poly", "toon", "stylized")):
                target_polycount = 40_000
            elif any(kw in prompt_lower for kw in _HUMANOID_KEYWORDS):
                target_polycount = 150_000
            elif asset_type == AssetType.ENVIRONMENT:
                target_polycount = 80_000
            else:
                target_polycount = 50_000

        if progress_callback:
            progress_callback(0.0, f"Starting generation ({asset_type.value} → {selected_model.value})...")
        
        # Stage 1: Preview
        if progress_callback:
            progress_callback(0.05, "Creating preview mesh...")
        
        preview = await self.text_to_3d_preview(
            prompt,
            art_style=art_style,
            ai_model=selected_model,
            pose_mode=pose_mode,
            target_polycount=target_polycount,
        )
        
        if progress_callback:
            progress_callback(0.1, f"Preview task submitted: {preview.id}")
        
        # Wait for preview with progress updates
        preview_result = await self._wait_with_progress(
            preview.id,
            "text-to-3d",
            timeout=timeout / 2,
            progress_callback=progress_callback,
            progress_range=(0.1, 0.45),
            phase_name="Generating mesh...",
        )
        
        # Stage 2: Refine (add textures)
        if progress_callback:
            progress_callback(0.5, "Adding textures...")
        
        refine = await self.text_to_3d_refine(
            preview_result.id,
            enable_pbr=enable_pbr,
            ai_model=selected_model,
        )
        
        # Wait for refine with progress updates
        final_result = await self._wait_with_progress(
            refine.id,
            "text-to-3d",
            timeout=timeout / 2,
            progress_callback=progress_callback,
            progress_range=(0.5, 0.95),
            phase_name="Generating textures...",
        )
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return final_result
    
    async def generate_animated_character(
        self,
        prompt: str,
        *,
        animations: Optional[List[str]] = None,
        ai_model: Optional[MeshyAIModel] = None,
        art_style: ArtStyle = ArtStyle.REALISTIC,
        enable_pbr: bool = True,
        height_meters: float = 1.7,
        target_polycount: int = DEFAULT_POLYCOUNT,
        timeout: float = 900.0,  # 15 minutes for full workflow
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete animated character from text.
        
        This is the full workflow for character generation:
        1. Generate 3D model (with A-pose for rigging)
        2. Rig the model (creates skeleton + basic walking/running animations)
        3. Generate additional animations if requested (via Animation Library API)
        
        Basic animations (walking, running) are included FREE with rigging.
        Additional animations (jump, attack, dance, etc.) require API calls.
        
        Args:
            prompt: Description of the character.
            animations: List of animation names to generate (e.g., ["walk", "run", "jump", "attack"]).
                       Walking and running use the free basic animations.
                       Other animations use the Animation Library API.
            ai_model: Override model selection (defaults to Meshy-5 for characters).
            art_style: Realistic or sculpture style.
            enable_pbr: Generate PBR texture maps.
            height_meters: Character height for rigging (default 1.7m).
            target_polycount: Target polygon count.
            timeout: Maximum total time for all stages.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Dictionary containing:
            - model_task: The final 3D model MeshyTask
            - rig_task: The rigging MeshyTask (with basic_animations)
            - animation_tasks: Dict mapping animation names to MeshyTask
            - animation_urls: Dict mapping animation names to download URLs
            
        Example:
            result = await client.generate_animated_character(
                "a medieval knight warrior",
                animations=["walk", "run", "sword slash", "death"]
            )
            
            # Walking is FREE (from rigging)
            walk_url = result["animation_urls"]["walk"]
            
            # Sword slash uses Animation Library API
            attack_url = result["animation_urls"]["sword slash"]
        """
        if animations is None:
            animations = ["walk", "run"]
        
        # Normalize animation names
        animations = [a.lower().strip() for a in animations]
        
        # Character generation uses LATEST (highest quality) - characters are essential assets
        selected_model = ai_model or MeshyAIModel.LATEST
        
        # Calculate time budget for each phase
        base_gen_time = timeout * 0.4  # 40% for generation
        rig_time = timeout * 0.3       # 30% for rigging
        anim_time = timeout * 0.3      # 30% for animations
        
        # Count how many API animations we need
        api_animations = [a for a in animations if not is_basic_animation(a)]
        
        if progress_callback:
            progress_callback(0.0, "Starting character generation...")
        
        # Stage 1: Generate 3D model with A-pose
        if progress_callback:
            progress_callback(0.02, "Generating 3D model...")
        
        model_task = await self.generate_3d_model(
            prompt,
            ai_model=selected_model,
            art_style=art_style,
            enable_pbr=enable_pbr,
            pose_mode=PoseMode.A_POSE,  # Required for rigging
            target_polycount=target_polycount,
            timeout=base_gen_time,
            progress_callback=lambda p, m: progress_callback(0.02 + p * 0.38, m) if progress_callback else None,
        )
        
        if progress_callback:
            progress_callback(0.40, "3D model complete, starting rigging...")
        
        # Stage 2: Rig the model
        rig_task = await self.rig_model(
            input_task_id=model_task.id,
            height_meters=height_meters,
            timeout=rig_time,
            progress_callback=lambda p, m: progress_callback(0.40 + p * 0.30, m) if progress_callback else None,
        )
        
        if progress_callback:
            progress_callback(0.70, f"Rigging complete! Processing {len(animations)} animations...")
        
        # Build result
        animation_tasks: Dict[str, MeshyTask] = {}
        animation_urls: Dict[str, str] = {}
        
        # Stage 3: Get animations
        # First, extract basic animations from rigging result
        if rig_task.basic_animations:
            if "walk" in animations or "walking" in animations:
                walk_url = rig_task.basic_animations.walking_glb_url or rig_task.basic_animations.walking_fbx_url
                if walk_url:
                    animation_urls["walk"] = walk_url
                    if "walking" in animations:
                        animation_urls["walking"] = walk_url
            
            if "run" in animations or "running" in animations:
                run_url = rig_task.basic_animations.running_glb_url or rig_task.basic_animations.running_fbx_url
                if run_url:
                    animation_urls["run"] = run_url
                    if "running" in animations:
                        animation_urls["running"] = run_url
        
        # Generate additional animations via API
        if api_animations:
            anim_time_per = anim_time / len(api_animations)
            
            for i, anim_name in enumerate(api_animations):
                action_id = get_animation_action_id(anim_name)
                
                if action_id is None:
                    # Unknown animation, skip
                    continue
                
                if progress_callback:
                    base_progress = 0.70 + (i / len(api_animations)) * 0.28
                    progress_callback(base_progress, f"Generating {anim_name} animation...")
                
                try:
                    anim_task = await self.animate_model(
                        rig_task.id,
                        action_id=action_id,
                        timeout=anim_time_per,
                        progress_callback=None,  # Don't spam progress for each animation
                    )
                    
                    animation_tasks[anim_name] = anim_task
                    anim_url = anim_task.animation_glb_url or anim_task.animation_fbx_url
                    if anim_url:
                        animation_urls[anim_name] = anim_url
                        
                except Exception as e:
                    # Log but don't fail the whole workflow
                    if progress_callback:
                        progress_callback(0.70, f"Warning: Failed to generate {anim_name}: {e}")
        
        if progress_callback:
            progress_callback(1.0, f"Character complete with {len(animation_urls)} animations!")
        
        return {
            "model_task": model_task,
            "rig_task": rig_task,
            "animation_tasks": animation_tasks,
            "animation_urls": animation_urls,
        }
    
    async def _wait_with_progress(
        self,
        task_id: str,
        task_type: str,
        timeout: float,
        progress_callback: Optional[Callable[[float, str], None]],
        progress_range: tuple[float, float] = (0.0, 1.0),
        phase_name: str = "Processing",
    ) -> MeshyTask:
        """Wait for task with progress updates mapped to a range."""
        start_time = time.time()
        start_progress, end_progress = progress_range
        
        while True:
            task = await self.get_task(task_id, task_type)
            
            if task.is_complete:
                if task.is_success:
                    return task
                else:
                    raise MeshyTaskError(
                        task_id=task_id,
                        status=task.status.value,
                        message=task.error_message,
                    )
            
            # Map task progress (0-100) to our range
            if progress_callback and task.progress > 0:
                mapped_progress = start_progress + (task.progress / 100.0) * (end_progress - start_progress)
                progress_callback(mapped_progress, phase_name)
            
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise asyncio.TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            await asyncio.sleep(self.POLL_INTERVAL)
    
    # =========================================================================
    # Text-to-3D API (Low-level)
    # =========================================================================
    
    async def text_to_3d_preview(
        self,
        prompt: str,
        *,
        art_style: ArtStyle = ArtStyle.REALISTIC,
        ai_model: MeshyAIModel = MeshyAIModel.LATEST,
        topology: Topology = Topology.TRIANGLE,
        target_polycount: int = DEFAULT_POLYCOUNT,
        symmetry_mode: SymmetryMode = SymmetryMode.AUTO,
        pose_mode: PoseMode = PoseMode.NONE,
        should_remesh: bool = True,
    ) -> MeshyTask:
        """
        Create Text-to-3D preview task (mesh only, no texture).
        First stage of the text-to-3D workflow.
        """
        if len(prompt) > 600:
            raise ValueError("Prompt must be <= 600 characters")
        
        target_polycount = max(self.MIN_POLYCOUNT, min(target_polycount, self.MAX_POLYCOUNT))
        
        body = {
            "mode": "preview",
            "prompt": prompt,
            "art_style": _enum_value(art_style),
            "ai_model": _enum_value(ai_model),
            "topology": _enum_value(topology),
            "target_polycount": target_polycount,
            "symmetry_mode": _enum_value(symmetry_mode),
            "pose_mode": _enum_value(pose_mode),
            "should_remesh": should_remesh,
        }
        
        client = self._get_client()
        response = await client.post(
            f"{self.BASE_URL}/openapi/{self.TEXT_TO_3D_VERSION}/text-to-3d",
            headers=self._headers,
            json=body,
        )
        response.raise_for_status()
        
        result = response.json()
        task_id = result.get("result")
        
        return MeshyTask(
            id=task_id,
            task_type="text-to-3d-preview",
            status=TaskStatus.PENDING,
            prompt=prompt,
        )
    
    async def text_to_3d_refine(
        self,
        preview_task_id: str,
        *,
        enable_pbr: bool = True,
        texture_prompt: Optional[str] = None,
        texture_image_url: Optional[str] = None,
        ai_model: MeshyAIModel = MeshyAIModel.LATEST,
    ) -> MeshyTask:
        """
        Create Text-to-3D refine task (adds textures to preview mesh).
        Second stage of the text-to-3D workflow.
        """
        body = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
            "enable_pbr": enable_pbr,
            "ai_model": _enum_value(ai_model),
        }
        
        if texture_prompt:
            body["texture_prompt"] = texture_prompt[:600]
        if texture_image_url:
            body["texture_image_url"] = texture_image_url
        
        client = self._get_client()
        response = await client.post(
            f"{self.BASE_URL}/openapi/{self.TEXT_TO_3D_VERSION}/text-to-3d",
            headers=self._headers,
            json=body,
        )
        response.raise_for_status()
        
        result = response.json()
        task_id = result.get("result")
        
        return MeshyTask(
            id=task_id,
            task_type="text-to-3d-refine",
            status=TaskStatus.PENDING,
        )
    
    async def get_text_to_3d_task(self, task_id: str) -> MeshyTask:
        """Get status of a Text-to-3D task."""
        client = self._get_client()
        response = await client.get(
            f"{self.BASE_URL}/openapi/{self.TEXT_TO_3D_VERSION}/text-to-3d/{task_id}",
            headers=self._headers,
        )
        response.raise_for_status()
        return self._parse_task(response.json())
    
    # =========================================================================
    # Image-to-3D API (STUB - Not yet implemented)
    # =========================================================================
    
    async def image_to_3d(
        self,
        image_url: str,
        *,
        ai_model: MeshyAIModel = MeshyAIModel.LATEST,
        topology: Topology = Topology.TRIANGLE,
        target_polycount: int = DEFAULT_POLYCOUNT,
        symmetry_mode: SymmetryMode = SymmetryMode.AUTO,
        should_remesh: bool = True,
        should_texture: bool = True,
        enable_pbr: bool = True,
        pose_mode: PoseMode = PoseMode.NONE,
        timeout: float = DEFAULT_TIMEOUT,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MeshyTask:
        """
        [STUB] Create Image-to-3D task from a single image.
        
        This function is a placeholder for future implementation.
        The API endpoint and parameters are documented but not yet integrated.
        
        Args:
            image_url: URL or base64 data URI of input image.
            ... (other parameters as per Meshy API docs)
            
        Raises:
            NotImplementedError: This feature is not yet implemented.
        """
        raise NotImplementedError(
            "Image-to-3D is not yet implemented. "
            "Use generate_3d_model() with a text prompt instead."
        )
    
    async def multi_image_to_3d(
        self,
        image_urls: List[str],
        *,
        ai_model: MeshyAIModel = MeshyAIModel.LATEST,
        enable_pbr: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MeshyTask:
        """
        [STUB] Create Multi-Image-to-3D task from 1-4 images of same object.
        
        This function is a placeholder for future implementation.
        
        Raises:
            NotImplementedError: This feature is not yet implemented.
        """
        raise NotImplementedError(
            "Multi-Image-to-3D is not yet implemented. "
            "Use generate_3d_model() with a text prompt instead."
        )
    
    # =========================================================================
    # Rigging API
    # =========================================================================
    
    async def rig_model(
        self,
        *,
        model_url: Optional[str] = None,
        input_task_id: Optional[str] = None,
        height_meters: float = 1.7,
        timeout: float = DEFAULT_TIMEOUT,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MeshyTask:
        """
        Rig a humanoid 3D model for animation.
        
        The model must be a humanoid character. After rigging, use animate_model()
        to apply animations from the Meshy animation library.
        
        Args:
            model_url: URL to a humanoid GLB model (required if input_task_id not provided).
            input_task_id: ID of a previous Meshy task (required if model_url not provided).
            height_meters: Character height for scaling (default 1.7m).
            timeout: Maximum wait time.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Completed MeshyTask with rigged model URLs.
            
        Example:
            # Rig a generated character
            character = await client.generate_3d_model("a knight", pose_mode=PoseMode.A_POSE)
            rigged = await client.rig_model(input_task_id=character.id)
            
            # Or rig from external URL
            rigged = await client.rig_model(model_url="https://example.com/character.glb")
        """
        if not model_url and not input_task_id:
            raise ValueError("Either model_url or input_task_id must be provided")
        
        if progress_callback:
            progress_callback(0.0, "Starting rigging...")
        
        body = {
            "height_meters": height_meters,
        }
        
        if model_url:
            body["model_url"] = model_url
        if input_task_id:
            body["input_task_id"] = input_task_id
        
        client = self._get_client()
        response = await client.post(
            f"{self.BASE_URL}/openapi/{self.RIGGING_VERSION}/rigging",
            headers=self._headers,
            json=body,
        )
        response.raise_for_status()
        
        result = response.json()
        task_id = result.get("result")
        
        if progress_callback:
            progress_callback(0.1, f"Rigging task submitted: {task_id}")
        
        # Wait for completion
        final_result = await self._wait_with_progress(
            task_id,
            "rigging",
            timeout=timeout,
            progress_callback=progress_callback,
            progress_range=(0.1, 0.95),
            phase_name="Rigging model...",
        )
        
        if progress_callback:
            progress_callback(1.0, "Rigging complete!")
        
        return final_result
    
    async def get_rigging_task(self, task_id: str) -> MeshyTask:
        """Get status of a rigging task."""
        client = self._get_client()
        response = await client.get(
            f"{self.BASE_URL}/openapi/{self.RIGGING_VERSION}/rigging/{task_id}",
            headers=self._headers,
        )
        response.raise_for_status()
        return self._parse_task(response.json())
    
    # =========================================================================
    # Animation API
    # =========================================================================
    
    async def animate_model(
        self,
        rig_task_id: str,
        action_id: int,
        *,
        fps: int = 30,
        timeout: float = DEFAULT_TIMEOUT,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MeshyTask:
        """
        Apply an animation to a rigged model.
        
        Args:
            rig_task_id: ID of a completed rigging task.
            action_id: Animation action ID from Meshy's animation library.
            fps: Target frames per second (default 30).
            timeout: Maximum wait time.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Completed MeshyTask with animated model URLs.
            
        Common Animation IDs (from Meshy library):
            - 92: Idle
            - 93: Walk
            - 94: Run
            - 95: Jump
            - 96: Attack
            (See Meshy documentation for full animation library)
            
        Example:
            rigged = await client.rig_model(input_task_id=character.id)
            walking = await client.animate_model(rigged.id, action_id=93)  # Walk animation
        """
        if progress_callback:
            progress_callback(0.0, "Starting animation...")
        
        body = {
            "rig_task_id": rig_task_id,
            "action_id": action_id,
            "post_process": {
                "operation_type": "change_fps",
                "fps": fps,
            }
        }
        
        client = self._get_client()
        response = await client.post(
            f"{self.BASE_URL}/openapi/{self.ANIMATION_VERSION}/animations",
            headers=self._headers,
            json=body,
        )
        response.raise_for_status()
        
        result = response.json()
        task_id = result.get("result")
        
        if progress_callback:
            progress_callback(0.1, f"Animation task submitted: {task_id}")
        
        # Wait for completion
        final_result = await self._wait_with_progress(
            task_id,
            "animation",
            timeout=timeout,
            progress_callback=progress_callback,
            progress_range=(0.1, 0.95),
            phase_name="Generating animation...",
        )
        
        if progress_callback:
            progress_callback(1.0, "Animation complete!")
        
        return final_result
    
    async def get_animation_task(self, task_id: str) -> MeshyTask:
        """Get status of an animation task."""
        client = self._get_client()
        response = await client.get(
            f"{self.BASE_URL}/openapi/{self.ANIMATION_VERSION}/animations/{task_id}",
            headers=self._headers,
        )
        response.raise_for_status()
        return self._parse_task(response.json())
    
    # =========================================================================
    # Retexture API
    # =========================================================================
    
    async def retexture_model(
        self,
        *,
        model_url: Optional[str] = None,
        input_task_id: Optional[str] = None,
        text_style_prompt: Optional[str] = None,
        image_style_url: Optional[str] = None,
        enable_original_uv: bool = True,
        enable_pbr: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MeshyTask:
        """
        Retexture an existing 3D model with a new style.
        
        Args:
            model_url: URL to the model (required if input_task_id not provided).
            input_task_id: ID of a previous Meshy task (required if model_url not provided).
            text_style_prompt: Text description of desired texture style.
            image_style_url: Reference image URL for style transfer.
            enable_original_uv: Preserve original UV mapping.
            enable_pbr: Generate PBR texture maps.
            timeout: Maximum wait time.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Completed MeshyTask with retextured model URLs.
            
        Example:
            # Retexture a model to look like gold
            retextured = await client.retexture_model(
                input_task_id=original.id,
                text_style_prompt="golden metallic surface with engravings"
            )
        """
        if not model_url and not input_task_id:
            raise ValueError("Either model_url or input_task_id must be provided")
        
        if progress_callback:
            progress_callback(0.0, "Starting retexture...")
        
        body = {
            "enable_original_uv": enable_original_uv,
            "enable_pbr": enable_pbr,
        }
        
        if model_url:
            body["model_url"] = model_url
        if input_task_id:
            body["input_task_id"] = input_task_id
        if text_style_prompt:
            body["text_style_prompt"] = text_style_prompt
        if image_style_url:
            body["image_style_url"] = image_style_url
        
        client = self._get_client()
        response = await client.post(
            f"{self.BASE_URL}/openapi/{self.RETEXTURE_VERSION}/retexture",
            headers=self._headers,
            json=body,
        )
        response.raise_for_status()
        
        result = response.json()
        task_id = result.get("result")
        
        if progress_callback:
            progress_callback(0.1, f"Retexture task submitted: {task_id}")
        
        # Wait for completion
        final_result = await self._wait_with_progress(
            task_id,
            "retexture",
            timeout=timeout,
            progress_callback=progress_callback,
            progress_range=(0.1, 0.95),
            phase_name="Retexturing model...",
        )
        
        if progress_callback:
            progress_callback(1.0, "Retexture complete!")
        
        return final_result
    
    async def get_retexture_task(self, task_id: str) -> MeshyTask:
        """Get status of a retexture task."""
        client = self._get_client()
        response = await client.get(
            f"{self.BASE_URL}/openapi/{self.RETEXTURE_VERSION}/retexture/{task_id}",
            headers=self._headers,
        )
        response.raise_for_status()
        return self._parse_task(response.json())
    
    # =========================================================================
    # Remesh API
    # =========================================================================
    
    async def remesh_model(
        self,
        *,
        model_url: Optional[str] = None,
        input_task_id: Optional[str] = None,
        target_formats: List[str] = None,
        topology: Topology = Topology.TRIANGLE,
        target_polycount: int = DEFAULT_POLYCOUNT,
        timeout: float = DEFAULT_TIMEOUT,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MeshyTask:
        """
        Remesh a model to optimize topology or convert formats.
        
        Args:
            model_url: URL to the model (required if input_task_id not provided).
            input_task_id: ID of a previous Meshy task.
            target_formats: List of formats to generate ["glb", "fbx", "obj", "usdz", "blend", "stl"].
            topology: Target mesh topology.
            target_polycount: Target polygon count.
            timeout: Maximum wait time.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Completed MeshyTask with remeshed model URLs.
        """
        if not model_url and not input_task_id:
            raise ValueError("Either model_url or input_task_id must be provided")
        
        if progress_callback:
            progress_callback(0.0, "Starting remesh...")
        
        body = {
            "topology": _enum_value(topology),
            "target_polycount": max(self.MIN_POLYCOUNT, min(target_polycount, self.MAX_POLYCOUNT)),
        }
        
        if model_url:
            body["model_url"] = model_url
        if input_task_id:
            body["input_task_id"] = input_task_id
        if target_formats:
            body["target_formats"] = target_formats
        
        client = self._get_client()
        response = await client.post(
            f"{self.BASE_URL}/openapi/{self.REMESH_VERSION}/remesh",
            headers=self._headers,
            json=body,
        )
        response.raise_for_status()
        
        result = response.json()
        task_id = result.get("result")
        
        if progress_callback:
            progress_callback(0.1, f"Remesh task submitted: {task_id}")
        
        # Wait for completion
        final_result = await self._wait_with_progress(
            task_id,
            "remesh",
            timeout=timeout,
            progress_callback=progress_callback,
            progress_range=(0.1, 0.95),
            phase_name="Remeshing model...",
        )
        
        if progress_callback:
            progress_callback(1.0, "Remesh complete!")
        
        return final_result
    
    async def get_remesh_task(self, task_id: str) -> MeshyTask:
        """Get status of a remesh task."""
        client = self._get_client()
        response = await client.get(
            f"{self.BASE_URL}/openapi/{self.REMESH_VERSION}/remesh/{task_id}",
            headers=self._headers,
        )
        response.raise_for_status()
        return self._parse_task(response.json())
    
    # =========================================================================
    # Text-to-Image API (STUB - Not yet implemented)
    # =========================================================================
    
    async def text_to_image(
        self,
        prompt: str,
        *,
        ai_model: str = "nano-banana",
        generate_multi_view: bool = False,
        aspect_ratio: str = "1:1",
        timeout: float = DEFAULT_TIMEOUT,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MeshyTask:
        """
        [STUB] Generate an image from text description.
        
        This function is a placeholder for future implementation.
        The API endpoint and parameters are documented but not yet integrated.
        
        Args:
            prompt: Text description of the image.
            ai_model: "nano-banana" or "nano-banana-pro".
            generate_multi_view: Generate multiple views for 3D conversion.
            aspect_ratio: Image aspect ratio ("1:1", "16:9", etc.).
            
        Raises:
            NotImplementedError: This feature is not yet implemented.
        """
        raise NotImplementedError(
            "Text-to-Image is not yet implemented. "
            "This feature is planned for future releases."
        )
    
    # =========================================================================
    # Task Status & Common Operations
    # =========================================================================
    
    async def get_task(self, task_id: str, task_type: str = "text-to-3d") -> MeshyTask:
        """
        Get task status by type.
        
        Args:
            task_id: Task ID to query.
            task_type: One of "text-to-3d", "image-to-3d", "rigging", "animation", "retexture", "remesh"
            
        Returns:
            MeshyTask with current status.
        """
        if task_type in ("text-to-3d", "text-to-3d-preview", "text-to-3d-refine"):
            return await self.get_text_to_3d_task(task_id)
        elif task_type == "rigging":
            return await self.get_rigging_task(task_id)
        elif task_type == "animation":
            return await self.get_animation_task(task_id)
        elif task_type == "retexture":
            return await self.get_retexture_task(task_id)
        elif task_type == "remesh":
            return await self.get_remesh_task(task_id)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def wait_for_completion(
        self,
        task_id: str,
        task_type: str = "text-to-3d",
        *,
        timeout: float = DEFAULT_TIMEOUT,
        poll_interval: float = POLL_INTERVAL,
    ) -> MeshyTask:
        """
        Wait for task to complete, polling at regular intervals.
        
        Consider using generate_3d_model() or other high-level methods
        which include progress callbacks instead.
        """
        start_time = time.time()
        
        while True:
            task = await self.get_task(task_id, task_type)
            
            if task.is_complete:
                if task.is_success:
                    return task
                else:
                    raise MeshyTaskError(
                        task_id=task_id,
                        status=task.status.value,
                        message=task.error_message,
                    )
            
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise asyncio.TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            await asyncio.sleep(poll_interval)
    
    async def download_model(
        self,
        url: str,
        output_path: Optional[str | Path] = None,
    ) -> bytes:
        """
        Download model file from result URL.
        
        Args:
            url: Download URL from completed task.
            output_path: Optional path to save the file.
            
        Returns:
            Model file bytes.
        """
        client = self._get_client()
        response = await client.get(url)
        response.raise_for_status()
        
        data = response.content
        
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        
        return data
    
    async def download_textures(
        self,
        task: MeshyTask,
        output_dir: str | Path,
    ) -> Dict[str, Path]:
        """
        Download all texture maps from a completed task.
        
        Args:
            task: Completed task with texture_urls.
            output_dir: Directory to save textures.
            
        Returns:
            Dict mapping texture type to saved file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved = {}
        
        for i, tex in enumerate(task.texture_urls):
            for tex_type in ["base_color", "metallic", "normal", "roughness"]:
                url = getattr(tex, tex_type)
                if url:
                    filename = f"texture_{i}_{tex_type}.png"
                    path = output_dir / filename
                    await self.download_model(url, path)
                    saved[f"{tex_type}_{i}"] = path
        
        return saved
    
    # =========================================================================
    # Webhook Handling
    # =========================================================================
    
    def set_webhook_callback(self, callback: Callable[[dict], None]) -> None:
        """
        Set a callback function to be called when webhook events are received.
        
        The callback will receive the raw webhook payload as a dictionary.
        
        Args:
            callback: Function to call with webhook event data.
        """
        self._webhook_callback = callback
    
    def validate_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """
        Validate a webhook signature from Meshy.
        
        Args:
            payload: Raw request body bytes.
            signature: Signature from X-Webhook-Signature header.
            
        Returns:
            True if signature is valid, False otherwise.
        """
        if not self.webhook_secret:
            return False
        
        import hmac
        import hashlib
        
        expected = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected, signature)
    
    def process_webhook_event(self, event: dict) -> None:
        """
        Process an incoming webhook event.
        
        This method is called by the webhook route handler when an event is received.
        It will invoke the registered callback if one is set.
        
        Args:
            event: The webhook event payload.
        """
        if self._webhook_callback:
            self._webhook_callback(event)


# =============================================================================
# Exceptions
# =============================================================================

class MeshyError(Exception):
    """Base exception for Meshy client."""
    pass


class MeshyAPIError(MeshyError):
    """API-level error from Meshy."""
    
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"[HTTP {status_code}] {message}")


class MeshyTaskError(MeshyError):
    """Task execution error."""
    
    def __init__(self, task_id: str, status: str, message: Optional[str]):
        self.task_id = task_id
        self.status = status
        self.message = message
        super().__init__(f"Task {task_id} {status}: {message}")


# =============================================================================
# Example Usage
# =============================================================================

async def example_generate_with_routing():
    """Example: Generate 3D models with automatic quality routing."""
    async with MeshyClient() as client:
        # Essential asset - will use Meshy-5
        print("Generating character (essential → Meshy-5)...")
        character = await client.generate_3d_model(
            "a knight warrior in full plate armor",
            pose_mode=PoseMode.A_POSE,  # For rigging
            progress_callback=lambda p, m: print(f"  [{p*100:.0f}%] {m}"),
        )
        print(f"Character GLB: {character.glb_url}")
        
        # Environment asset - will use Meshy-5
        print("\nGenerating environment prop (environment → Meshy-5)...")
        barrel = await client.generate_3d_model(
            "a wooden barrel with metal bands",
            progress_callback=lambda p, m: print(f"  [{p*100:.0f}%] {m}"),
        )
        print(f"Barrel GLB: {barrel.glb_url}")


async def example_rig_and_animate():
    """Example: Generate a character, rig it, and add animation."""
    async with MeshyClient() as client:
        # Generate character
        print("Generating character...")
        character = await client.generate_3d_model(
            "a robot warrior",
            pose_mode=PoseMode.A_POSE,
            progress_callback=lambda p, m: print(f"  [{p*100:.0f}%] {m}"),
        )
        
        # Rig the character
        print("\nRigging character...")
        rigged = await client.rig_model(
            input_task_id=character.id,
            progress_callback=lambda p, m: print(f"  [{p*100:.0f}%] {m}"),
        )
        print(f"Rigged model: {rigged.glb_url}")
        
        # Add walk animation
        print("\nAdding walk animation...")
        animated = await client.animate_model(
            rigged.id,
            action_id=93,  # Walk
            progress_callback=lambda p, m: print(f"  [{p*100:.0f}%] {m}"),
        )
        print(f"Animated model: {animated.glb_url}")


if __name__ == "__main__":
    asyncio.run(example_generate_with_routing())
