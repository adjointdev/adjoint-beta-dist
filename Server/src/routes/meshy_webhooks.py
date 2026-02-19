"""
Meshy Webhook Route Handler

Handles incoming webhook events from Meshy.ai for real-time task status updates.
This allows the server to receive push notifications instead of polling for task status.

Webhook events are sent when:
- Task status changes (PENDING → IN_PROGRESS → SUCCEEDED/FAILED)
- Task completes
- Task fails

Setup:
1. Configure WEBHOOK_URL in .env with your public HTTPS endpoint
2. Configure WEBHOOK_SECRET in .env for signature validation
3. Register the webhook URL in Meshy API settings dashboard
"""

import hmac
import hashlib
import logging
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

logger = logging.getLogger("adjoint-server")

# In-memory storage for pending task callbacks
# Maps task_id -> callback function
_task_callbacks: Dict[str, Callable[[dict], None]] = {}

# Storage for completed task results (for polling fallback)
# Maps task_id -> result dict
_task_results: Dict[str, dict] = {}


def register_task_callback(task_id: str, callback: Callable[[dict], None]) -> None:
    """
    Register a callback to be invoked when a webhook event is received for this task.
    
    Args:
        task_id: The Meshy task ID to monitor.
        callback: Function to call with the webhook event payload.
    """
    _task_callbacks[task_id] = callback
    logger.debug(f"[Webhook] Registered callback for task {task_id}")


def unregister_task_callback(task_id: str) -> None:
    """Remove a task callback registration."""
    _task_callbacks.pop(task_id, None)


def get_task_result(task_id: str) -> Optional[dict]:
    """Get stored result for a completed task."""
    return _task_results.get(task_id)


def clear_task_result(task_id: str) -> None:
    """Clear stored result for a task."""
    _task_results.pop(task_id, None)


def validate_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str
) -> bool:
    """
    Validate the webhook signature from Meshy.
    
    Meshy uses HMAC-SHA256 for webhook signature verification.
    
    Args:
        payload: Raw request body bytes.
        signature: Signature from request header.
        secret: Webhook secret from configuration.
        
    Returns:
        True if signature is valid.
    """
    if not secret:
        logger.warning("[Webhook] No webhook secret configured - skipping validation")
        return True  # Allow in development
    
    expected = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected, signature)


async def handle_meshy_webhook(request: Request) -> Response:
    """
    Handle incoming Meshy webhook events.
    
    Expected payload format:
    {
        "event": "task.succeeded" | "task.failed" | "task.started",
        "task_id": "018a210d-8ba4-705c-b111-1f1776f7f578",
        "task_type": "text-to-3d" | "image-to-3d" | "rigging" | "animation",
        "status": "SUCCEEDED" | "FAILED" | "IN_PROGRESS",
        "progress": 100,
        "model_urls": { ... },
        "created_at": "2024-01-01T00:00:00Z",
        "finished_at": "2024-01-01T00:05:00Z"
    }
    """
    import os
    
    try:
        # Read raw body for signature validation
        body = await request.body()
        
        # Validate signature if configured
        webhook_secret = os.getenv("WEBHOOK_SECRET", "")
        signature = request.headers.get("X-Webhook-Signature", "")
        
        if webhook_secret and signature:
            if not validate_webhook_signature(body, signature, webhook_secret):
                logger.warning("[Webhook] Invalid signature - rejecting request")
                return JSONResponse(
                    {"error": "Invalid signature"},
                    status_code=401
                )
        
        # Parse payload
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            logger.error(f"[Webhook] Invalid JSON payload: {e}")
            return JSONResponse(
                {"error": "Invalid JSON"},
                status_code=400
            )
        
        # Extract task info
        event_type = payload.get("event", "unknown")
        task_id = payload.get("task_id") or payload.get("id")
        task_type = payload.get("task_type") or payload.get("type", "unknown")
        status = payload.get("status", "unknown")
        progress = payload.get("progress", 0)
        
        logger.info(
            f"[Webhook] Received {event_type} for task {task_id} "
            f"(type={task_type}, status={status}, progress={progress}%)"
        )
        
        # Store result for polling fallback
        if task_id:
            _task_results[task_id] = {
                "event": event_type,
                "task_id": task_id,
                "task_type": task_type,
                "status": status,
                "progress": progress,
                "model_urls": payload.get("model_urls"),
                "texture_urls": payload.get("texture_urls"),
                "thumbnail_url": payload.get("thumbnail_url"),
                "error_message": payload.get("task_error", {}).get("message"),
                "received_at": datetime.utcnow().isoformat(),
            }
            
            # Invoke callback if registered
            callback = _task_callbacks.get(task_id)
            if callback:
                try:
                    callback(payload)
                    logger.debug(f"[Webhook] Callback invoked for task {task_id}")
                except Exception as e:
                    logger.error(f"[Webhook] Callback error for task {task_id}: {e}")
                
                # Clean up callback after terminal states
                if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                    unregister_task_callback(task_id)
        
        # Respond with 200 OK to acknowledge receipt
        return JSONResponse(
            {"status": "received", "task_id": task_id},
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"[Webhook] Error processing webhook: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


async def get_webhook_status(request: Request) -> Response:
    """
    Debug endpoint to check webhook handler status.
    
    GET /webhooks/meshy/status
    
    Returns info about pending callbacks and recent results.
    """
    return JSONResponse({
        "pending_callbacks": list(_task_callbacks.keys()),
        "stored_results": len(_task_results),
        "recent_task_ids": list(_task_results.keys())[-10:],
    })


async def get_task_webhook_result(request: Request) -> Response:
    """
    Get stored webhook result for a task (polling fallback).
    
    GET /webhooks/meshy/task/{task_id}
    
    Returns the stored result or 404 if not found.
    """
    task_id = request.path_params.get("task_id")
    
    result = get_task_result(task_id)
    if result:
        return JSONResponse(result)
    else:
        return JSONResponse(
            {"error": "Task not found", "task_id": task_id},
            status_code=404
        )


# Route definitions for FastMCP/Starlette
webhook_routes = [
    Route("/webhooks/meshy", handle_meshy_webhook, methods=["POST"]),
    Route("/webhooks/meshy/status", get_webhook_status, methods=["GET"]),
    Route("/webhooks/meshy/task/{task_id}", get_task_webhook_result, methods=["GET"]),
]


def get_webhook_routes():
    """Get the list of webhook routes to register with the server."""
    return webhook_routes
