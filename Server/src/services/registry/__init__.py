"""
Registry package for MCP tool auto-discovery.
"""
from .tool_registry import (
    adjoint_tool,
    get_registered_tools,
    clear_tool_registry,
)
from .resource_registry import (
    adjoint_resource,
    get_registered_resources,
    clear_resource_registry,
)

__all__ = [
    'adjoint_tool',
    'get_registered_tools',
    'clear_tool_registry',
    'adjoint_resource',
    'get_registered_resources',
    'clear_resource_registry'
]
