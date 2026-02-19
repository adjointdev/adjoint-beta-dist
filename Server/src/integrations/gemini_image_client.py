"""
Google Gemini Imagen API Client for Image Generation

This client provides async access to Google's Imagen API for image generation:
- 2D texture generation (for 3D models, materials)
- Sprite generation (game characters, items, UI elements)
- Sprite sheet generation (animation frames)
- General image generation

API Documentation: https://ai.google.dev/gemini-api/docs/image-generation

Environment Variables:
    GEMINI_API_KEY: Your Google AI API key

Usage:
    from integrations.gemini_image_client import GeminiImageClient

    async with GeminiImageClient() as client:
        # Generate a texture
        result = await client.generate_texture(
            prompt="seamless cobblestone texture, photorealistic",
            width=512,
            height=512
        )
        
        # Generate a sprite
        result = await client.generate_sprite(
            prompt="pixel art knight character, side view",
            style="pixel_art",
            width=64,
            height=64
        )
"""

import os
import asyncio
import logging
import base64
from typing import Optional, Literal, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import httpx

logger = logging.getLogger("adjoint-server.gemini_image")


class GeminiImageError(Exception):
    """Base exception for Gemini Image API errors."""
    pass


class GeminiImageAPIError(GeminiImageError):
    """Raised when the Gemini API returns an error."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Gemini Image API error ({status_code}): {message}")


class GeminiImageConfigError(GeminiImageError):
    """Raised when configuration is missing or invalid."""
    pass


class ImageStyle(str, Enum):
    """Predefined styles for image generation."""
    PHOTOREALISTIC = "photorealistic"
    DIGITAL_ART = "digital_art"
    PIXEL_ART = "pixel_art"
    CARTOON = "cartoon"
    ANIME = "anime"
    WATERCOLOR = "watercolor"
    OIL_PAINTING = "oil_painting"
    SKETCH = "sketch"
    FLAT = "flat"
    LOW_POLY = "low_poly"


class ImageType(str, Enum):
    """Types of images that can be generated."""
    TEXTURE = "texture"
    SPRITE = "sprite"
    SPRITE_SHEET = "sprite_sheet"
    ICON = "icon"
    UI_ELEMENT = "ui_element"
    BACKGROUND = "background"
    CONCEPT_ART = "concept_art"
    GENERAL = "general"


class AspectRatio(str, Enum):
    """Supported aspect ratios for image generation."""
    SQUARE_1_1 = "1:1"
    LANDSCAPE_16_9 = "16:9"
    LANDSCAPE_4_3 = "4:3"
    PORTRAIT_9_16 = "9:16"
    PORTRAIT_3_4 = "3:4"


@dataclass
class ImageResult:
    """Result of an image generation request."""
    image_data: bytes
    """Raw image bytes (PNG format)."""
    
    width: int
    """Image width in pixels."""
    
    height: int
    """Image height in pixels."""
    
    prompt: str
    """The prompt used for generation."""
    
    style: Optional[str] = None
    """Style applied to the generation."""
    
    image_type: str = "general"
    """Type of image generated."""
    
    mime_type: str = "image/png"
    """MIME type of the image."""
    
    def save(self, path: str) -> str:
        """Save the image to a file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(self.image_data)
        logger.info(f"Saved image to {path}")
        return path
    
    def to_base64(self) -> str:
        """Convert image to base64 string."""
        return base64.b64encode(self.image_data).decode('utf-8')


@dataclass
class SpriteSheetResult:
    """Result of a sprite sheet generation request."""
    image_data: bytes
    """Raw sprite sheet image bytes (PNG format)."""
    
    width: int
    """Total sprite sheet width in pixels."""
    
    height: int
    """Total sprite sheet height in pixels."""
    
    frame_width: int
    """Width of each frame in pixels."""
    
    frame_height: int
    """Height of each frame in pixels."""
    
    frame_count: int
    """Number of frames in the sprite sheet."""
    
    columns: int
    """Number of columns in the sprite sheet grid."""
    
    rows: int
    """Number of rows in the sprite sheet grid."""
    
    prompt: str
    """The prompt used for generation."""
    
    style: Optional[str] = None
    """Style applied to the generation."""
    
    mime_type: str = "image/png"
    """MIME type of the image."""
    
    def save(self, path: str) -> str:
        """Save the sprite sheet to a file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(self.image_data)
        logger.info(f"Saved sprite sheet to {path} ({self.frame_count} frames, {self.columns}x{self.rows} grid)")
        return path


# Style prompt modifiers for different image types
STYLE_PROMPTS = {
    ImageStyle.PHOTOREALISTIC: "photorealistic, highly detailed, professional photography",
    ImageStyle.DIGITAL_ART: "digital art, vibrant colors, detailed illustration",
    ImageStyle.PIXEL_ART: "pixel art style, retro game aesthetic, limited color palette, crisp pixels",
    ImageStyle.CARTOON: "cartoon style, bold outlines, bright colors, stylized",
    ImageStyle.ANIME: "anime style, cel shaded, Japanese animation aesthetic",
    ImageStyle.WATERCOLOR: "watercolor painting, soft edges, artistic brush strokes",
    ImageStyle.OIL_PAINTING: "oil painting, textured brush strokes, classical art style",
    ImageStyle.SKETCH: "pencil sketch, hand-drawn, artistic lines",
    ImageStyle.FLAT: "flat design, minimal shadows, solid colors, modern vector style",
    ImageStyle.LOW_POLY: "low poly 3D render style, geometric shapes, faceted surfaces",
}

# Type-specific prompt enhancements
TYPE_PROMPTS = {
    ImageType.TEXTURE: "seamless tileable texture, game asset, uniform lighting, no shadows",
    ImageType.SPRITE: "game sprite, transparent background, clean edges, suitable for 2D game",
    ImageType.SPRITE_SHEET: "animation sprite sheet, consistent character, uniform frame size",
    ImageType.ICON: "game icon, clear readable design, suitable for UI",
    ImageType.UI_ELEMENT: "game UI element, clean design, suitable for interface",
    ImageType.BACKGROUND: "game background, suitable for scrolling or static backdrop",
    ImageType.CONCEPT_ART: "concept art, detailed design exploration",
    ImageType.GENERAL: "",
}


class GeminiImageClient:
    """
    Async client for Google Imagen API image generation.
    
    Uses Imagen 4 model for high-quality image generation.
    
    Supports context manager for automatic cleanup:
        async with GeminiImageClient() as client:
            result = await client.generate_image(...)
    """
    
    # Imagen API endpoint
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    # Imagen 4 - latest and best quality image generation model
    MODEL = "imagen-4.0-generate-001"
    # Alternative models:
    # - "imagen-4.0-fast-generate-001" - Faster but slightly lower quality
    # - "imagen-4.0-ultra-generate-001" - Highest quality (may be slower/more expensive)
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        model: Optional[str] = None
    ):
        """
        Initialize the Gemini Image client.
        
        Args:
            api_key: Google AI API key. If not provided, reads from GEMINI_API_KEY env var.
            timeout: Request timeout in seconds.
            model: Model to use. Defaults to imagen-4.0-generate-001.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise GeminiImageConfigError(
                "GEMINI_API_KEY environment variable not set. "
                "Get your API key from https://aistudio.google.com/app/apikey"
            )
        
        self.timeout = timeout
        self.model = model or self.MODEL
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "GeminiImageClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    def _build_prompt(
        self,
        prompt: str,
        image_type: ImageType = ImageType.GENERAL,
        style: Optional[ImageStyle] = None,
        additional_modifiers: Optional[str] = None
    ) -> str:
        """Build an enhanced prompt with style and type modifiers."""
        parts = [prompt]
        
        # Add style modifier
        if style and style in STYLE_PROMPTS:
            parts.append(STYLE_PROMPTS[style])
        
        # Add type modifier
        if image_type in TYPE_PROMPTS and TYPE_PROMPTS[image_type]:
            parts.append(TYPE_PROMPTS[image_type])
        
        # Add custom modifiers
        if additional_modifiers:
            parts.append(additional_modifiers)
        
        return ", ".join(parts)
    
    def _get_aspect_ratio(self, width: int, height: int) -> str:
        """Calculate the closest supported aspect ratio."""
        ratio = width / height
        
        if 0.95 <= ratio <= 1.05:
            return "1:1"
        elif ratio > 1.5:
            return "16:9"
        elif ratio > 1.2:
            return "4:3"
        elif ratio < 0.67:
            return "9:16"
        elif ratio < 0.85:
            return "3:4"
        else:
            return "1:1"
    
    async def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        style: Optional[ImageStyle] = None,
        image_type: ImageType = ImageType.GENERAL,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> ImageResult:
        """
        Generate an image using Google Imagen 4.
        
        Args:
            prompt: Text description of the image to generate.
            width: Requested image width (actual size may vary based on aspect ratio).
            height: Requested image height (actual size may vary based on aspect ratio).
            style: Optional style preset to apply.
            image_type: Type of image (affects prompt enhancement).
            negative_prompt: Things to avoid in the generation.
            num_images: Number of images to generate (returns first).
            progress_callback: Optional callback for progress updates.
        
        Returns:
            ImageResult with the generated image data.
        """
        client = self._get_client()
        
        # Build enhanced prompt
        enhanced_prompt = self._build_prompt(prompt, image_type, style)
        
        # Add negative prompt guidance
        if negative_prompt:
            enhanced_prompt += f". Avoid: {negative_prompt}"
        
        if progress_callback:
            progress_callback(0.1, "Preparing image generation request...")
        
        logger.info(f"Generating image with Imagen 4: {enhanced_prompt[:100]}...")
        
        # Build request payload for Imagen predict endpoint
        url = f"{self.BASE_URL}/models/{self.model}:predict"
        
        payload = {
            "instances": [
                {"prompt": enhanced_prompt}
            ],
            "parameters": {
                "sampleCount": num_images,
                "aspectRatio": self._get_aspect_ratio(width, height)
            }
        }
        
        if progress_callback:
            progress_callback(0.2, "Sending request to Imagen 4...")
        
        try:
            response = await client.post(
                url,
                json=payload,
                params={"key": self.api_key}
            )
            
            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_detail = error_json["error"].get("message", response.text)
                except:
                    pass
                raise GeminiImageAPIError(response.status_code, error_detail)
            
            if progress_callback:
                progress_callback(0.8, "Processing generated image...")
            
            result = response.json()
            
            # Extract image data from response
            # Response structure: predictions[0].bytesBase64Encoded
            if "predictions" not in result or len(result["predictions"]) == 0:
                raise GeminiImageError("No predictions returned from API")
            
            prediction = result["predictions"][0]
            
            if "bytesBase64Encoded" not in prediction:
                raise GeminiImageError("No image data in response")
            
            image_data = base64.b64decode(prediction["bytesBase64Encoded"])
            mime_type = prediction.get("mimeType", "image/png")
            
            if progress_callback:
                progress_callback(1.0, "Image generation complete!")
            
            return ImageResult(
                image_data=image_data,
                width=width,
                height=height,
                prompt=prompt,
                style=style.value if style else None,
                image_type=image_type.value,
                mime_type=mime_type
            )
            
        except httpx.TimeoutException:
            raise GeminiImageError("Request timed out. Try again or use a simpler prompt.")
        except httpx.RequestError as e:
            raise GeminiImageError(f"Network error: {str(e)}")
    
    async def generate_texture(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        style: Optional[ImageStyle] = None,
        seamless: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> ImageResult:
        """
        Generate a texture suitable for 3D models or materials.
        
        Args:
            prompt: Description of the texture (e.g., "rusty metal", "wooden planks").
            width: Texture width in pixels.
            height: Texture height in pixels.
            style: Optional style preset.
            seamless: If True, generates a tileable/seamless texture.
            progress_callback: Optional callback for progress updates.
        
        Returns:
            ImageResult with the generated texture.
        """
        # Enhance prompt for texture generation
        texture_modifiers = []
        if seamless:
            texture_modifiers.append("seamless tileable pattern")
        texture_modifiers.append("high quality texture map")
        texture_modifiers.append("uniform lighting, no harsh shadows")
        texture_modifiers.append("suitable for 3D model UV mapping")
        
        enhanced_prompt = f"{prompt}, {', '.join(texture_modifiers)}"
        
        return await self.generate_image(
            prompt=enhanced_prompt,
            width=width,
            height=height,
            style=style,
            image_type=ImageType.TEXTURE,
            negative_prompt="text, watermark, logo, border, frame, signature",
            progress_callback=progress_callback
        )
    
    async def generate_sprite(
        self,
        prompt: str,
        width: int = 256,
        height: int = 256,
        style: ImageStyle = ImageStyle.PIXEL_ART,
        transparent_background: bool = True,
        view: str = "front",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> ImageResult:
        """
        Generate a 2D sprite for games.
        
        Args:
            prompt: Description of the sprite (e.g., "knight character", "treasure chest").
            width: Sprite width in pixels.
            height: Sprite height in pixels.
            style: Art style (default: pixel_art).
            transparent_background: If True, requests transparent background.
            view: Character view angle ("front", "side", "back", "top-down", "isometric").
            progress_callback: Optional callback for progress updates.
        
        Returns:
            ImageResult with the generated sprite.
        """
        sprite_modifiers = [f"{view} view"]
        
        if transparent_background:
            sprite_modifiers.append("transparent background")
            sprite_modifiers.append("isolated on transparent")
        
        sprite_modifiers.append("clean edges")
        sprite_modifiers.append("game-ready sprite asset")
        sprite_modifiers.append("single character or object")
        
        enhanced_prompt = f"{prompt}, {', '.join(sprite_modifiers)}"
        
        return await self.generate_image(
            prompt=enhanced_prompt,
            width=width,
            height=height,
            style=style,
            image_type=ImageType.SPRITE,
            negative_prompt="multiple characters, background scenery, text, watermark, blurry edges",
            progress_callback=progress_callback
        )
    
    async def generate_sprite_sheet(
        self,
        prompt: str,
        frames: int = 4,
        frame_width: int = 64,
        frame_height: int = 64,
        animation_type: str = "walk_cycle",
        style: ImageStyle = ImageStyle.PIXEL_ART,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> SpriteSheetResult:
        """
        Generate an animated sprite sheet.
        
        Args:
            prompt: Description of the character/object to animate.
            frames: Number of animation frames (4, 6, 8 recommended).
            frame_width: Width of each frame in pixels.
            frame_height: Height of each frame in pixels.
            animation_type: Type of animation ("walk_cycle", "idle", "attack", "run", "jump").
            style: Art style (default: pixel_art).
            progress_callback: Optional callback for progress updates.
        
        Returns:
            SpriteSheetResult with the generated sprite sheet.
        """
        # Calculate grid layout
        if frames <= 4:
            columns = frames
            rows = 1
        elif frames <= 8:
            columns = 4
            rows = 2
        else:
            columns = 4
            rows = (frames + 3) // 4
        
        total_width = columns * frame_width
        total_height = rows * frame_height
        
        # Build sprite sheet specific prompt
        animation_descriptions = {
            "walk_cycle": "walking animation sequence showing leg and arm movement",
            "idle": "idle breathing animation with subtle movement",
            "attack": "attack swing animation sequence",
            "run": "running animation with dynamic pose changes",
            "jump": "jump animation from crouch to apex to landing",
        }
        
        anim_desc = animation_descriptions.get(animation_type, animation_type)
        
        sprite_sheet_prompt = (
            f"{prompt}, sprite sheet format, {frames} animation frames arranged horizontally, "
            f"{anim_desc}, consistent character appearance across all frames, "
            f"uniform frame size, clear frame separation, game asset sprite sheet, "
            f"transparent background between frames"
        )
        
        if progress_callback:
            progress_callback(0.1, f"Generating {frames}-frame sprite sheet...")
        
        result = await self.generate_image(
            prompt=sprite_sheet_prompt,
            width=total_width,
            height=total_height,
            style=style,
            image_type=ImageType.SPRITE_SHEET,
            negative_prompt="overlapping frames, inconsistent character, blurry, text, watermark",
            progress_callback=progress_callback
        )
        
        return SpriteSheetResult(
            image_data=result.image_data,
            width=total_width,
            height=total_height,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_count=frames,
            columns=columns,
            rows=rows,
            prompt=prompt,
            style=style.value if style else None
        )
    
    async def generate_icon(
        self,
        prompt: str,
        size: int = 128,
        style: ImageStyle = ImageStyle.FLAT,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> ImageResult:
        """
        Generate a game icon (for UI, inventory, abilities, etc.).
        
        Args:
            prompt: Description of the icon (e.g., "health potion", "sword", "fire spell").
            size: Icon size in pixels (square).
            style: Art style (default: flat).
            progress_callback: Optional callback for progress updates.
        
        Returns:
            ImageResult with the generated icon.
        """
        icon_prompt = (
            f"{prompt}, game icon, centered composition, clear silhouette, "
            f"readable at small size, suitable for inventory or UI"
        )
        
        return await self.generate_image(
            prompt=icon_prompt,
            width=size,
            height=size,
            style=style,
            image_type=ImageType.ICON,
            negative_prompt="text, complex background, tiny details, multiple objects",
            progress_callback=progress_callback
        )
    
    async def generate_background(
        self,
        prompt: str,
        width: int = 1920,
        height: int = 1080,
        style: Optional[ImageStyle] = None,
        parallax_layer: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> ImageResult:
        """
        Generate a game background image.
        
        Args:
            prompt: Description of the background scene.
            width: Background width in pixels.
            height: Background height in pixels.
            style: Art style.
            parallax_layer: For parallax scrolling ("foreground", "midground", "background", "sky").
            progress_callback: Optional callback for progress updates.
        
        Returns:
            ImageResult with the generated background.
        """
        bg_modifiers = ["game background", "scenic", "atmospheric"]
        
        if parallax_layer:
            layer_descriptions = {
                "foreground": "foreground layer with detailed close objects, some transparency",
                "midground": "middle ground layer with medium distance objects",
                "background": "background layer with distant scenery",
                "sky": "sky layer with clouds or celestial objects"
            }
            bg_modifiers.append(layer_descriptions.get(parallax_layer, parallax_layer))
        
        enhanced_prompt = f"{prompt}, {', '.join(bg_modifiers)}"
        
        return await self.generate_image(
            prompt=enhanced_prompt,
            width=width,
            height=height,
            style=style,
            image_type=ImageType.BACKGROUND,
            negative_prompt="text, watermark, UI elements, characters in focus",
            progress_callback=progress_callback
        )


async def download_image(url: str, output_path: str) -> str:
    """
    Download an image from a URL to a local file.
    
    Args:
        url: The URL of the image to download.
        output_path: Local path to save the image.
    
    Returns:
        The output path where the image was saved.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded image to {output_path}")
        return output_path
