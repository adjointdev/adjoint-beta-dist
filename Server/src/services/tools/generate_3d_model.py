"""
Generate 3D Model Tool for Adjoint

This tool provides AI-powered 3D model generation capabilities for Unity:
- Text-to-3D: Generate models from text descriptions
- Image-to-3D: Generate models from reference images

Providers:
- Hunyuan3D (Primary): Best quality, cheapest, for static props/environments
- Meshy.ai (Secondary): Auto-rigging support, for characters

Usage via Adjoint:
    # Generate a sword from text
    await generate_3d_model(
        action="generate",
        prompt="a medieval sword with ornate handle",
        provider="hunyuan3d"
    )
    
    # Generate a rigged character from image
    await generate_3d_model(
        action="generate",
        image_url="https://example.com/character.png",
        provider="meshy",
        for_rigging=True
    )
"""

import os
import asyncio
from typing import Annotated, Any, Literal, Optional
from pathlib import Path

from fastmcp import Context
from services.registry import adjoint_tool
from services.tools import get_unity_instance_from_context
from transport.unity_transport import send_with_unity_instance


# Import our 3D generation clients
try:
    from integrations.hunyuan3d_client import (
        Hunyuan3DClient,
        GenerateType,
        Hunyuan3DTaskError,
    )
    HUNYUAN3D_AVAILABLE = True
except ImportError:
    HUNYUAN3D_AVAILABLE = False

try:
    from integrations.meshy_client import (
        MeshyClient,
        MeshyAIModel,
        PoseMode,
        ArtStyle,
        MeshyTaskError,
        get_model_for_asset,
    )
    MESHY_AVAILABLE = True
except ImportError:
    MESHY_AVAILABLE = False


# --- Smart polycount defaults ---
_POLYCOUNT_BY_TYPE = {
    "character": 150_000, "humanoid": 150_000, "creature": 150_000, "npc": 150_000,
    "environment": 80_000,
    "prop": 50_000,
}
_LOW_POLY_KEYWORDS = ["low poly", "lowpoly", "low-poly", "toon", "stylized"]


def _get_default_polycount(prompt: Optional[str], asset_type: str) -> int:
    """Return a sensible polycount based on prompt keywords and asset type."""
    if prompt and any(kw in prompt.lower() for kw in _LOW_POLY_KEYWORDS):
        return 40_000
    return _POLYCOUNT_BY_TYPE.get((asset_type or "prop").lower(), 50_000)


# Provider selection logic
def _get_recommended_provider(
    prompt: Optional[str],
    image_url: Optional[str],
    for_rigging: bool,
    asset_type: str,
) -> str:
    """
    Determine the best provider based on use case.
    
    Decision Matrix:
    - Characters with rigging → Meshy (auto-rigging support)
    - Static props/weapons → Hunyuan3D (best quality)
    - Environment objects → Hunyuan3D (self-hosted option)
    - Quick prototypes → Meshy or Hunyuan3D based on availability
    """
    if for_rigging:
        return "meshy"  # Only Meshy has auto-rigging
    
    asset_type_lower = asset_type.lower() if asset_type else ""
    
    # Character-related assets benefit from Meshy's rigging
    character_keywords = ["character", "humanoid", "person", "creature", "animal", "npc"]
    if any(kw in asset_type_lower for kw in character_keywords):
        return "meshy"
    
    # Props and environment → Hunyuan3D for quality
    return "hunyuan3d"


@adjoint_tool(
    description="""Generate 3D models using AI (Hunyuan3D or Meshy.ai) and import them into Unity.

Providers:
- hunyuan3d: Best quality for static props and environments (Tencent)
- meshy: Best for characters that need rigging/animation

Actions:
- generate: Create a new 3D model from text or image
- status: Check generation status
- import: Import a previously generated model

Examples:
- Generate sword: action="generate", prompt="medieval sword", provider="hunyuan3d"
- Generate character: action="generate", prompt="robot warrior", provider="meshy", for_rigging=True
- From image: action="generate", image_url="https://...", provider="meshy"
"""
)
async def generate_3d_model(
    ctx: Context,
    action: Annotated[
        Literal["generate", "status", "import", "list_providers"],
        "Action to perform: generate new model, check status, import existing, or list providers"
    ],
    prompt: Annotated[
        Optional[str],
        "Text description for text-to-3D generation (max 600 chars for Meshy, 1024 for Hunyuan3D)"
    ] = None,
    image_url: Annotated[
        Optional[str],
        "Image URL or base64 data URI for image-to-3D generation"
    ] = None,
    provider: Annotated[
        Optional[Literal["hunyuan3d", "meshy", "auto"]],
        "AI provider to use. 'auto' selects based on use case (default)"
    ] = "auto",
    asset_name: Annotated[
        Optional[str],
        "Name for the generated asset in Unity (auto-generated if not provided)"
    ] = None,
    asset_type: Annotated[
        Optional[str],
        "Type hint for the asset (e.g., 'character', 'prop', 'environment')"
    ] = "prop",
    for_rigging: Annotated[
        bool,
        "Generate in A-pose for rigging (Meshy only, enables pose_mode)"
    ] = False,
    enable_pbr: Annotated[
        bool,
        "Generate PBR texture maps (metallic, roughness, normal)"
    ] = True,
    target_polycount: Annotated[
        Optional[int],
        "Target polygon count (default: 30000 for Meshy, 500000 for Hunyuan3D)"
    ] = None,
    import_path: Annotated[
        str,
        "Unity Assets subfolder for imported models"
    ] = "GeneratedModels",
    import_immediately: Annotated[
        bool,
        "Import to Unity immediately after generation completes"
    ] = True,
    timeout: Annotated[
        float,
        "Maximum wait time for generation in seconds"
    ] = 600.0,
    # Status/import specific
    task_id: Annotated[
        Optional[str],
        "Task ID for status check or import (from previous generate call)"
    ] = None,
    task_provider: Annotated[
        Optional[str],
        "Provider of the task (for status/import)"
    ] = None,
) -> dict[str, Any]:
    """Generate or import AI-generated 3D models into Unity."""
    
    unity_instance = get_unity_instance_from_context(ctx)
    
    # Action: List available providers
    if action == "list_providers":
        providers = []
        if HUNYUAN3D_AVAILABLE:
            has_creds = bool(os.getenv("TENCENT_SECRET_ID") or os.getenv("SecretId"))
            providers.append({
                "id": "hunyuan3d",
                "name": "Tencent Hunyuan3D 2.5",
                "available": has_creds,
                "best_for": ["props", "environments", "high-volume"],
                "features": ["text-to-3d", "image-to-3d", "multi-view", "pbr"],
                "missing": None if has_creds else "TENCENT_SECRET_ID and TENCENT_SECRET_KEY env vars"
            })
        if MESHY_AVAILABLE:
            has_creds = bool(os.getenv("MESHY_API_KEY"))
            providers.append({
                "id": "meshy",
                "name": "Meshy.ai",
                "available": has_creds,
                "best_for": ["characters", "rigging", "animation"],
                "features": ["text-to-3d", "image-to-3d", "auto-rigging", "a-pose", "pbr"],
                "missing": None if has_creds else "MESHY_API_KEY env var"
            })
        return {
            "success": True,
            "providers": providers,
            "recommendation": "Use 'auto' provider selection for best results"
        }
    
    # Action: Check status
    if action == "status":
        if not task_id:
            return {"success": False, "message": "task_id required for status check"}
        if not task_provider:
            return {"success": False, "message": "task_provider required for status check"}
        
        return await _check_task_status(ctx, task_id, task_provider)
    
    # Action: Import existing
    if action == "import":
        if not task_id:
            return {"success": False, "message": "task_id required for import"}
        if not task_provider:
            return {"success": False, "message": "task_provider required for import"}
        
        return await _import_model(
            ctx, unity_instance, task_id, task_provider,
            asset_name, import_path, enable_pbr, asset_type
        )
    
    # Action: Generate new model
    if action == "generate":
        if not prompt and not image_url:
            return {
                "success": False,
                "message": "Either 'prompt' (text-to-3D) or 'image_url' (image-to-3D) is required"
            }
        
        # Auto-select provider
        if provider == "auto":
            provider = _get_recommended_provider(prompt, image_url, for_rigging, asset_type)
            await ctx.info(f"Auto-selected provider: {provider}")
        
        # Generate asset name if not provided
        if not asset_name:
            if prompt:
                # Use first few words of prompt
                words = prompt.split()[:3]
                asset_name = "_".join(words).replace(" ", "_")[:32]
            else:
                asset_name = f"model_{int(asyncio.get_event_loop().time())}"
        
        # Dispatch to provider
        if provider == "hunyuan3d":
            return await _generate_with_hunyuan3d(
                ctx, unity_instance, prompt, image_url,
                asset_name, import_path, enable_pbr, target_polycount,
                import_immediately, timeout
            )
        elif provider == "meshy":
            return await _generate_with_meshy(
                ctx, unity_instance, prompt, image_url,
                asset_name, import_path, enable_pbr, target_polycount,
                for_rigging, import_immediately, timeout, asset_type
            )
        else:
            return {"success": False, "message": f"Unknown provider: {provider}"}
    
    return {"success": False, "message": f"Unknown action: {action}"}


async def _generate_with_hunyuan3d(
    ctx: Context,
    unity_instance,
    prompt: Optional[str],
    image_url: Optional[str],
    asset_name: str,
    import_path: str,
    enable_pbr: bool,
    target_polycount: Optional[int],
    import_immediately: bool,
    timeout: float,
) -> dict[str, Any]:
    """Generate model using Hunyuan3D."""
    
    if not HUNYUAN3D_AVAILABLE:
        return {
            "success": False,
            "message": "Hunyuan3D client not available. Check integrations module."
        }
    
    try:
        async with Hunyuan3DClient() as client:
            await ctx.info(f"Starting Hunyuan3D generation: {prompt or image_url[:50]}")
            
            # Submit task
            if prompt:
                task = await client.text_to_3d(
                    prompt,
                    generate_type=GenerateType.NORMAL,
                    face_count=target_polycount or 500_000,
                    enable_pbr=enable_pbr,
                )
            else:
                task = await client.image_to_3d(
                    image_url=image_url,
                    generate_type=GenerateType.NORMAL,
                    face_count=target_polycount or 500_000,
                    enable_pbr=enable_pbr,
                )
            
            await ctx.info(f"Task submitted: {task.job_id}")
            
            # Wait for completion
            result = await client.wait_for_completion(
                task.job_id,
                timeout=timeout,
                poll_interval=5.0,
            )
            
            await ctx.info(f"Generation complete: {result.status}")
            
            # Import to Unity if requested
            if import_immediately and result.glb_url:
                import_result = await _import_to_unity(
                    ctx, unity_instance,
                    result.glb_url, asset_name, import_path,
                    "hunyuan3d", task.job_id
                )
                return {
                    "success": True,
                    "provider": "hunyuan3d",
                    "task_id": task.job_id,
                    "status": "completed",
                    "model_url": result.glb_url,
                    "imported": True,
                    "asset_path": import_result.get("asset_path"),
                    "message": f"Model generated and imported to {import_result.get('asset_path')}"
                }
            
            return {
                "success": True,
                "provider": "hunyuan3d",
                "task_id": task.job_id,
                "status": "completed",
                "model_url": result.glb_url,
                "imported": False,
                "message": "Model generated. Use action='import' to import to Unity."
            }
            
    except Hunyuan3DTaskError as e:
        return {
            "success": False,
            "provider": "hunyuan3d",
            "task_id": e.job_id,
            "error": str(e),
            "message": f"Generation failed: {e.error_message}"
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "provider": "hunyuan3d",
            "error": "timeout",
            "message": f"Generation timed out after {timeout}s"
        }
    except Exception as e:
        await ctx.error(f"Hunyuan3D error: {e}")
        return {
            "success": False,
            "provider": "hunyuan3d",
            "error": str(e),
            "message": f"Generation failed: {e}"
        }


async def _generate_with_meshy(
    ctx: Context,
    unity_instance,
    prompt: Optional[str],
    image_url: Optional[str],
    asset_name: str,
    import_path: str,
    enable_pbr: bool,
    target_polycount: Optional[int],
    for_rigging: bool,
    import_immediately: bool,
    timeout: float,
    asset_type: str = "prop",
) -> dict[str, Any]:
    """Generate model using Meshy.ai."""
    
    if not MESHY_AVAILABLE:
        return {
            "success": False,
            "message": "Meshy client not available. Check integrations module."
        }
    
    try:
        async with MeshyClient() as client:
            await ctx.info(f"Starting Meshy generation: {prompt or image_url[:50]}")

            pose_mode = PoseMode.A_POSE if for_rigging else PoseMode.NONE
            polycount = target_polycount or _get_default_polycount(prompt, asset_type)

            # Determine which Meshy model to use (routing currently forces Meshy-5)
            selected_model = get_model_for_asset(prompt) if prompt else MeshyAIModel.MESHY_5
            await ctx.info(f"Using Meshy model: {selected_model.value}")

            if prompt:
                # Text-to-3D: Two-stage workflow
                await ctx.info("Stage 1: Generating preview mesh...")
                preview = await client.text_to_3d_preview(
                    prompt,
                    art_style=ArtStyle.REALISTIC,
                    ai_model=selected_model,
                    pose_mode=pose_mode,
                    target_polycount=polycount,
                )

                preview_result = await client.wait_for_completion(
                    preview.id,
                    task_type="text-to-3d",
                    timeout=timeout / 2,
                )

                await ctx.info("Stage 2: Adding textures...")
                refine = await client.text_to_3d_refine(
                    preview_result.id,
                    enable_pbr=enable_pbr,
                    ai_model=selected_model,
                )

                result = await client.wait_for_completion(
                    refine.id,
                    task_type="text-to-3d",
                    timeout=timeout / 2,
                )
                task_id = refine.id

            else:
                # Image-to-3D: Single stage
                task = await client.image_to_3d(
                    image_url=image_url,
                    ai_model=selected_model,
                    enable_pbr=enable_pbr,
                    pose_mode=pose_mode,
                    target_polycount=polycount,
                )

                result = await client.wait_for_completion(
                    task.id,
                    task_type="image-to-3d",
                    timeout=timeout,
                )
                task_id = task.id
            
            await ctx.info(f"Generation complete: {result.status}")
            
            # Import to Unity if requested
            if import_immediately and result.glb_url:
                import_result = await _import_to_unity(
                    ctx, unity_instance,
                    result.glb_url, asset_name, import_path,
                    "meshy", task_id
                )
                
                response = {
                    "success": True,
                    "provider": "meshy",
                    "meshy_model": selected_model.value,
                    "task_id": task_id,
                    "status": "completed",
                    "model_urls": {
                        "glb": result.glb_url,
                        "fbx": result.fbx_url,
                    },
                    "imported": True,
                    "asset_path": import_result.get("asset_path"),
                    "message": f"Model generated and imported to {import_result.get('asset_path')}"
                }

                if for_rigging:
                    response["rigging_ready"] = True
                    response["rigging_tip"] = "Model is in A-pose. Upload to Mixamo for auto-rigging."

                return response

            return {
                "success": True,
                "provider": "meshy",
                "meshy_model": selected_model.value,
                "task_id": task_id,
                "status": "completed",
                "model_urls": {
                    "glb": result.glb_url,
                    "fbx": result.fbx_url,
                },
                "imported": False,
                "message": "Model generated. Use action='import' to import to Unity."
            }
            
    except MeshyTaskError as e:
        return {
            "success": False,
            "provider": "meshy",
            "task_id": e.task_id,
            "error": str(e),
            "message": f"Generation failed: {e.message}"
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "provider": "meshy",
            "error": "timeout",
            "message": f"Generation timed out after {timeout}s"
        }
    except Exception as e:
        await ctx.error(f"Meshy error: {e}")
        return {
            "success": False,
            "provider": "meshy",
            "error": str(e),
            "message": f"Generation failed: {e}"
        }


async def _check_task_status(
    ctx: Context,
    task_id: str,
    provider: str,
) -> dict[str, Any]:
    """Check status of a generation task."""
    
    if provider == "hunyuan3d":
        if not HUNYUAN3D_AVAILABLE:
            return {"success": False, "message": "Hunyuan3D not available"}
        
        async with Hunyuan3DClient() as client:
            task = await client.get_task_status(task_id)
            return {
                "success": True,
                "provider": "hunyuan3d",
                "task_id": task_id,
                "status": task.status.value,
                "is_complete": task.is_complete,
                "is_success": task.is_success,
                "model_url": task.glb_url,
                "error": task.error_message,
            }
    
    elif provider == "meshy":
        if not MESHY_AVAILABLE:
            return {"success": False, "message": "Meshy not available"}
        
        async with MeshyClient() as client:
            # Try text-to-3d first, then image-to-3d
            try:
                task = await client.get_text_to_3d_task(task_id)
            except Exception:
                task = await client.get_image_to_3d_task(task_id)
            
            return {
                "success": True,
                "provider": "meshy",
                "task_id": task_id,
                "status": task.status.value,
                "progress": task.progress,
                "is_complete": task.is_complete,
                "is_success": task.is_success,
                "model_url": task.glb_url,
                "error": task.error_message,
            }
    
    return {"success": False, "message": f"Unknown provider: {provider}"}


async def _import_model(
    ctx: Context,
    unity_instance,
    task_id: str,
    provider: str,
    asset_name: Optional[str],
    import_path: str,
    enable_pbr: bool,
    asset_type: str,
) -> dict[str, Any]:
    """Import a completed model to Unity."""
    
    # Get model URL from task
    status = await _check_task_status(ctx, task_id, provider)
    
    if not status.get("success"):
        return status
    
    if not status.get("is_success"):
        return {
            "success": False,
            "message": f"Task not completed: {status.get('status')}",
            "status": status
        }
    
    model_url = status.get("model_url")
    if not model_url:
        return {"success": False, "message": "No model URL available"}
    
    if not asset_name:
        asset_name = f"model_{task_id[:8]}"
    
    return await _import_to_unity(
        ctx, unity_instance,
        model_url, asset_name, import_path,
        provider, task_id
    )


async def _import_to_unity(
    ctx: Context,
    unity_instance,
    model_url: str,
    asset_name: str,
    import_path: str,
    provider: str,
    task_id: str,
) -> dict[str, Any]:
    """Import model from URL to Unity using ModelImportService."""
    
    await ctx.info(f"Importing model to Unity: {asset_name}")
    
    # Call Unity's ModelImportService via transport
    command = {
        "action": "import_3d_model",
        "model_url": model_url,
        "asset_name": asset_name,
        "import_path": import_path,
        "provider": provider,
        "task_id": task_id,
    }
    
    try:
        result = await send_with_unity_instance(
            unity_instance,
            "ModelImport",
            command
        )
        
        if result.get("success"):
            return {
                "success": True,
                "asset_path": result.get("asset_path"),
                "message": f"Model imported to {result.get('asset_path')}"
            }
        else:
            return {
                "success": False,
                "message": result.get("message", "Import failed"),
                "error": result.get("error")
            }
    except Exception as e:
        await ctx.error(f"Unity import error: {e}")
        return {
            "success": False,
            "message": f"Failed to import to Unity: {e}",
            "model_url": model_url,
            "tip": "You can manually download from model_url and import to Unity"
        }
