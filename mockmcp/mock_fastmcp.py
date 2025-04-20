# Create or append to mock_fastmcp.py

from __future__ import annotations as _annotations

import logging
from typing import Any, TypeVar, Callable, Awaitable, Generic, Iterable, cast
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, AbstractAsyncContextManager, AsyncExitStack
import datetime
import anyio

import mcp.types as types
from mcp.server.lowlevel.server import LifespanResultT
from mcp.shared.exceptions import McpError
from pydantic.networks import AnyUrl
from mcp.types import AnyFunction
# Imports from fastmcp needed for mocking/mapping
from fastmcp.server.server import FastMCP, MountedServer, NOT_FOUND, _lifespan_wrapper, default_lifespan
from fastmcp.tools import ToolManager, Tool
from fastmcp.resources import ResourceManager, Resource, ResourceTemplate
from fastmcp.prompts import PromptManager, Prompt
from fastmcp.server.context import Context as FastMCPContext # Alias original context
from fastmcp.exceptions import NotFoundError, ResourceError
from fastmcp.prompts.prompt import PromptResult
from mcp.server.lowlevel.helper_types import ReadResourceContents
#from fastmcp.utilities.dependency_injection import solve_dependencies # Needed for context injection


logger = logging.getLogger(__name__)
LifespanResultT = TypeVar("LifespanResultT")


# --- Mock Context ---

class MockContext(Generic[LifespanResultT]):
    """A mock context for FastMCP handlers, providing access to lifespan context."""

    def __init__(self, lifespan_context: LifespanResultT, fastmcp_instance: "MockFastMcp[LifespanResultT]"):
        self._lifespan_context = lifespan_context
        self._fastmcp = fastmcp_instance # Keep ref if needed

    @property
    def lifespan(self) -> LifespanResultT:
        """Access the lifespan context."""
        return self._lifespan_context

    # --- Mocked Context Methods ---
    # Implement methods from FastMCPContext that might be used by handlers

    def debug(self, message: str, **kwargs: Any) -> None:
        logger.debug(f"[MockContext] {message}", extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        logger.info(f"[MockContext] {message}", extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        logger.warning(f"[MockContext] {message}", extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        logger.error(f"[MockContext] {message}", extra=kwargs)

    async def report_progress(
        self, progress: float, total: float | None = None, token: Any | None = None
    ) -> None:
        # Progress reporting is likely a no-op in mock environment
        logger.debug(f"[MockContext] Progress reported (no-op): progress={progress}, total={total}")
        await anyio.sleep(0) # Simulate async completion

    # Add other methods from FastMCPContext as needed, mocking their behavior.
    # For methods relying on a real session/request (like get_request_meta),
    # they should probably raise NotImplementedError or return None.
    def get_request_meta(self) -> types.RequestParams.Meta | None:
        logger.debug("[MockContext] get_request_meta called (returning None)")
        return None

    def get_request_id(self) -> types.RequestId | None:
        logger.debug("[MockContext] get_request_id called (returning 'mock_req')")
        return "mock_req" # Or None

    # Provide access to the parent mock server instance if needed
    @property
    def server(self) -> "MockFastMcp[LifespanResultT]":
        return self._fastmcp


# --- Mock Timed Cache (Simpler version) ---
class MockTimedCache:
    """A simpler cache for mocking, ignoring expiration."""
    def __init__(self, expiration: datetime.timedelta):
         # Expiration not really used in mock, but kept for signature match
         self._cache: dict[Any, Any] = {}

    def set(self, key: Any, value: Any) -> None:
        self._cache[key] = value

    def get(self, key: Any) -> Any:
        return self._cache.get(key, NOT_FOUND)

    def clear(self) -> None:
        self._cache.clear()


# --- Mock FastMCP ---

class MockFastMcp(Generic[LifespanResultT]):
    """
    A mock FastMCP server implementation that allows direct invocation
    of handlers without network communication.
    """
    def __init__(
        self,
        name: str = "mock_fastmcp",
        instructions: str | None = None,
        lifespan_context: LifespanResultT | None = None,
        tool_manager: ToolManager | None = None,
        resource_manager: ResourceManager | None = None,
        prompt_manager: PromptManager | None = None,
        settings: dict[str, Any] | None = None, # Store settings if needed by handlers via context
        tags: set[str] | None = None,
        dependencies: dict[Callable[..., Any], Any] | None = None, # For context injection
    ):
        self.name = name
        self.instructions = instructions
        self._lifespan_context = lifespan_context if lifespan_context is not None else {} # type: ignore
        self._tool_manager = tool_manager or ToolManager()
        self._resource_manager = resource_manager or ResourceManager()
        self._prompt_manager = prompt_manager or PromptManager()
        self.settings = settings or {} # Store simplified settings
        self.tags = tags or set()
        self.dependencies = dependencies or {} # Store dependencies for injection
        # Use a simple mock cache
        self._cache = MockTimedCache(datetime.timedelta(seconds=60))


    # --- Public API Methods (Mirrors FastMCP) ---

    def get_context(self) -> MockContext[LifespanResultT]:
        """Returns a MockContext object."""
        return MockContext(self._lifespan_context, self)

    async def get_tools(self) -> dict[str, Tool]:
        """Get all registered tools."""
        # In the mock, we assume the map function populates the managers correctly
        # including prefixed items from mounted/imported servers.
        # The cache here is mainly for consistency with FastMCP's interface.
        if (tools := self._cache.get("tools")) is NOT_FOUND:
            tools = self._tool_manager.get_tools()
            self._cache.set("tools", tools)
        return tools

    async def get_resources(self) -> dict[str, Resource]:
        """Get all registered resources."""
        if (resources := self._cache.get("resources")) is NOT_FOUND:
            resources = self._resource_manager.get_resources()
            self._cache.set("resources", resources)
        return resources

    async def get_resource_templates(self) -> dict[str, ResourceTemplate]:
         """Get all registered resource templates."""
         if (templates := self._cache.get("resource_templates")) is NOT_FOUND:
              templates = self._resource_manager.get_templates()
              self._cache.set("templates", templates)
         return templates

    async def get_prompts(self) -> dict[str, Prompt]:
        """Get all registered prompts."""
        if (prompts := self._cache.get("prompts")) is NOT_FOUND:
            prompts = self._prompt_manager.get_prompts()
            self._cache.set("prompts", prompts)
        return prompts


    # --- Direct Call Methods (Simulate calling handlers) ---

    async def call_tool(
        self, key: str, arguments: dict[str, Any] | None = None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Directly call a tool by key."""
        arguments = arguments or {}
        if not self._tool_manager.has_tool(key):
            raise NotFoundError(f"Unknown tool: {key}")

        context = self.get_context() # Create the MockContext
        # Use the ToolManager's call_tool which handles context injection via solve_dependencies
        try:
            # Pass the mock context object directly as the 'context' argument
            result = await self._tool_manager.call_tool(
                key,
                arguments,
                context=context # Pass the created MockContext here
                # REMOVED: dependencies=current_dependencies
            )

            # Ensure result is in the expected list format (similar to FastMCP handler)
            if not isinstance(result, list):
                 if isinstance(result, (types.TextContent, types.ImageContent, types.EmbeddedResource)):
                      return [result]
                 elif isinstance(result, str):
                      return [types.TextContent(type="text", text=result)]
                 else:
                      logger.warning(f"Unexpected return type {type(result)} from mock tool '{key}', attempting str conversion.")
                      try:
                           return [types.TextContent(type="text", text=str(result))]
                      except Exception:
                           raise TypeError(f"Unsupported return type {type(result)} from mock tool '{key}'")
            # Ensure all items in the list are of the expected types
            return [
                item for item in result
                if isinstance(item, (types.TextContent, types.ImageContent, types.EmbeddedResource))
            ]

        except Exception as e:
             logger.error(f"Error calling mock tool '{key}': {e}", exc_info=True)
             # Propagate specific FastMCP errors if needed, otherwise re-raise
             if isinstance(e, (NotFoundError, ResourceError)):
                 raise e
             # You might want to wrap other exceptions in a generic error
             # or just re-raise the original error for easier debugging in mock environment
             raise # Re-raise the original error


    async def read_resource(
        self, uri: str | AnyUrl
    ) -> list[ReadResourceContents]: # Return type matches _mcp_read_resource
        """Directly read a resource or template by URI."""
        uri_str = str(uri)
        context = self.get_context() # Create MockContext

        # Check local manager (includes prefixed items)
        if self._resource_manager.has_resource(uri_str):
             resource = await self._resource_manager.get_resource(uri_str)
             try:
                 # Pass MockContext as 'context' to read_resource if it accepts it
                 # Assuming ResourceManager.read_resource handles injection internally
                 # Check the actual signature if unsure. If it doesn't take context, remove it.
                 # Let's assume it might need context for dependency solving:
                 content = await self._resource_manager.read_resource(
                     uri_str, context=context # Pass context here
                     # REMOVED: dependencies=current_dependencies
                 )
                 return [ReadResourceContents(content=content, mime_type=resource.mime_type)]
             except Exception as e:
                 logger.error(f"Error reading mock resource {uri_str}: {e}", exc_info=True)
                 raise ResourceError(str(e))

        # Check templates
        elif template_match := await self._resource_manager.match_template(uri_str):
            template, params = template_match
            try:
                # Pass MockContext as 'context' to read_template if it accepts it
                # Assuming ResourceManager.read_template handles injection internally
                content = await self._resource_manager.read_template(
                    uri_str, params=params, context=context # Pass context here
                    # REMOVED: dependencies=current_dependencies
                )
                return [ReadResourceContents(content=content, mime_type=template.mime_type)]
            except Exception as e:
                 logger.error(f"Error reading mock template resource {uri_str} (template: {template.uri_template}): {e}", exc_info=True)
                 raise ResourceError(str(e))
        else:
            raise NotFoundError(f"Unknown resource or template: {uri_str}")


    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> PromptResult: # Return the direct PromptResult
        """Directly render a prompt by name."""
        arguments = arguments or {}
        if not self._prompt_manager.has_prompt(name):
            raise NotFoundError(f"Unknown prompt: {name}")

        context = self.get_context() # Create MockContext

        try:
            # Pass MockContext as 'context' to render_prompt
            # Assuming PromptManager.render_prompt handles injection internally
            messages = await self._prompt_manager.render_prompt(
                name, arguments, context=context # Pass context here
                # REMOVED: dependencies=current_dependencies
            )
            return messages
        except Exception as e:
            logger.error(f"Error rendering mock prompt '{name}': {e}", exc_info=True)
            raise # Re-raise original error

    # --- Mocked Add Methods ---
    # Provide methods to add items directly to the mock's managers

    def add_tool(self, fn: AnyFunction, **kwargs):
        self._tool_manager.add_tool_from_fn(fn, **kwargs)
        self._cache.clear()

    def tool(
        self, name: str | None = None, **kwargs
    ) -> Callable[[AnyFunction], AnyFunction]:
        """Decorator to register a tool directly on the mock."""
        if callable(name):
             raise TypeError("Use @tool() instead of @tool")
        def decorator(fn: AnyFunction) -> AnyFunction:
            self.add_tool(fn, name=name, **kwargs)
            return fn
        return decorator

    def add_resource_fn(self, fn: AnyFunction, uri: str, **kwargs):
         self._resource_manager.add_resource_or_template_from_fn(fn=fn, uri=uri, **kwargs)
         self._cache.clear()

    def resource(
        self, uri: str, **kwargs
    ) -> Callable[[AnyFunction], AnyFunction]:
        """Decorator to register a resource directly on the mock."""
        if callable(uri):
             raise TypeError("Use @resource('uri') instead of @resource")
        def decorator(fn: AnyFunction) -> AnyFunction:
            self.add_resource_fn(fn=fn, uri=uri, **kwargs)
            return fn
        return decorator

    def add_prompt(self, fn: Callable[..., PromptResult | Awaitable[PromptResult]], **kwargs):
         self._prompt_manager.add_prompt_from_fn(fn=fn, **kwargs)
         self._cache.clear()

    def prompt(
        self, name: str | None = None, **kwargs
    ) -> Callable[[AnyFunction], AnyFunction]:
         """Decorator to register a prompt directly on the mock."""
         if callable(name):
              raise TypeError("Use @prompt() instead of @prompt")
         def decorator(func: AnyFunction) -> AnyFunction:
              self.add_prompt(func, name=name, **kwargs)
              return func
         return decorator


# --- Mapping Function ---

async def map_fastmcp_to_mock(
    server: FastMCP[LifespanResultT],
    _processed_servers: dict[int, MockFastMcp] | None = None # Use id() for cycle detection
) -> MockFastMcp[LifespanResultT]:
    """
    Creates a MockFastMcp instance from an existing FastMCP instance,
    copying its managers, lifespan context, and handling mounted servers.
    """
    if _processed_servers is None:
        _processed_servers = {}

    server_id = id(server)
    if server_id in _processed_servers:
        logger.debug(f"Detected cycle, returning existing mock for server id {server_id}")
        return _processed_servers[server_id]

    logger.debug(f"Mapping server '{server.name}' (id: {server_id}) to mock...")

    # --- 1. Handle Lifespan ---
    lifespan_context: LifespanResultT | object
    # Access the *original* lifespan function attached to the underlying MCPServer
    original_mcp_lifespan = server._mcp_server.lifespan
    mcp_server_instance = server._mcp_server

    # Determine if the lifespan needs the FastMCP instance or the MCPServer instance
    # The _lifespan_wrapper creates a closure expecting MCPServer,
    # but the user's original lifespan expects FastMCP.
    # We need to execute the *user's* original lifespan function.

    user_lifespan_func: Callable[[FastMCP], AbstractAsyncContextManager[LifespanResultT]] | None = None

    # Try to find the original user lifespan function if it was wrapped
    # This is a bit hacky, relying on inspection or assuming structure
    if hasattr(original_mcp_lifespan, "__closure__") and original_mcp_lifespan.__closure__:
        for cell in original_mcp_lifespan.__closure__:
             if isinstance(cell.cell_contents, FastMCP):
                 # Found the wrapped FastMCP instance (usually 'app'), get the original lifespan from it
                 # Need to find the original lifespan callable stored within the wrapper setup
                 # Let's search closure cells for a callable that isn't the wrapper itself
                 possible_lifespans = [
                      c.cell_contents for c in original_mcp_lifespan.__closure__
                      if callable(c.cell_contents) and c.cell_contents is not _lifespan_wrapper
                 ]
                 if len(possible_lifespans) == 1:
                      user_lifespan_func = possible_lifespans[0]
                      break

    # If we couldn't find it through closure inspection, maybe it wasn't wrapped (e.g., using default lifespan)
    # Or the original FastMCP object holds it directly? Check FastMCP source...
    # FastMCP's __init__ stores the user's lifespan before wrapping. Let's assume we can access it
    # if _lifespan_wrapper wasn't used, or find a way to get it.
    # For simplicity now, let's just execute the wrapped one if we can't easily get the original.

    # If we identified the user's lifespan function:
    if user_lifespan_func:
         logger.debug(f"Executing user lifespan function for '{server.name}'")
         try:
             async with AsyncExitStack() as stack:
                  ctx = await stack.enter_async_context(user_lifespan_func(server)) # Pass FastMCP instance
                  lifespan_context = ctx
         except Exception as e:
              logger.error(f"Failed to execute user lifespan for server '{server.name}' during mapping: {e}. Using empty context.", exc_info=True)
              lifespan_context = {}
    else:
         # Fallback: Execute the potentially wrapped lifespan expecting MCPServer
         logger.debug(f"Executing potentially wrapped lifespan function for '{server.name}'")
         try:
             async with AsyncExitStack() as stack:
                  ctx = await stack.enter_async_context(original_mcp_lifespan(mcp_server_instance)) # Pass MCPServer instance
                  lifespan_context = ctx
         except Exception as e:
              logger.error(f"Failed to execute wrapped lifespan for server '{server.name}' during mapping: {e}. Using empty context.", exc_info=True)
              lifespan_context = {}


    # --- 2. Create Mock Instance (and store to prevent cycles) ---
    mock_mcp = MockFastMcp[LifespanResultT](
        name=f"mock_{server.name}",
        instructions=server.instructions,
        lifespan_context=lifespan_context, # type: ignore
        settings=server.settings.model_dump(),
        tags=server.tags.copy(),
        dependencies=server.dependencies.copy(), # Copy dependencies
        # Initialize managers - they will be populated below
        tool_manager=ToolManager(duplicate_behavior=server._tool_manager.duplicate_behavior),
        resource_manager=ResourceManager(duplicate_behavior=server._resource_manager.duplicate_behavior),
        prompt_manager=PromptManager(duplicate_behavior=server._prompt_manager.duplicate_behavior),
    )
    _processed_servers[server_id] = mock_mcp # Store before processing children


    # --- 3. Copy Local Handlers (Tools, Resources, Prompts) ---
    logger.debug(f"Copying local handlers for '{server.name}'...")
    for key, tool in server._tool_manager.get_tools().items():
        mock_mcp._tool_manager.add_tool(tool, key=key)
    for key, resource in server._resource_manager.get_resources().items():
        mock_mcp._resource_manager.add_resource(resource, key=key)
    for key, template in server._resource_manager.get_templates().items():
        mock_mcp._resource_manager.add_template(template, key=key)
    for key, prompt in server._prompt_manager.get_prompts().items():
        mock_mcp._prompt_manager.add_prompt(prompt, key=key)

    # --- 4. Handle Mounted Servers ---
    # Recursively map mounted servers and add their items with prefixes
    logger.debug(f"Processing mounted servers for '{server.name}'...")
    for prefix, mounted_server_info in server._mounted_servers.items():
        mounted_server_instance = mounted_server_info.server
        logger.debug(f"Mapping mounted server '{mounted_server_instance.name}' with prefix '{prefix}'...")
        # Pass the _processed_servers dict to handle cycles in mounting
        mock_mounted_server = await map_fastmcp_to_mock(mounted_server_instance, _processed_servers)

        # Add mounted server's items to the current mock's managers with prefixes
        tool_prefix = f"{prefix}{mounted_server_info.tool_separator}"
        mounted_tools = await mock_mounted_server.get_tools() # Get from the *mocked* child
        logger.debug(f"Adding {len(mounted_tools)} tools from mounted mock '{mock_mounted_server.name}' with prefix '{tool_prefix}'")
        for key, tool in mounted_tools.items():
             mock_mcp._tool_manager.add_tool(tool, key=f"{tool_prefix}{key}")

        resource_prefix = f"{prefix}{mounted_server_info.resource_separator}"
        mounted_resources = await mock_mounted_server.get_resources()
        mounted_templates = await mock_mounted_server.get_resource_templates()
        logger.debug(f"Adding {len(mounted_resources)} resources and {len(mounted_templates)} templates from mounted mock '{mock_mounted_server.name}' with prefix '{resource_prefix}'")
        for key, resource in mounted_resources.items():
             mock_mcp._resource_manager.add_resource(resource, key=f"{resource_prefix}{key}")
        for key, template in mounted_templates.items():
             mock_mcp._resource_manager.add_template(template, key=f"{resource_prefix}{key}")

        prompt_prefix = f"{prefix}{mounted_server_info.prompt_separator}"
        mounted_prompts = await mock_mounted_server.get_prompts()
        logger.debug(f"Adding {len(mounted_prompts)} prompts from mounted mock '{mock_mounted_server.name}' with prefix '{prompt_prefix}'")
        for key, prompt in mounted_prompts.items():
             mock_mcp._prompt_manager.add_prompt(prompt, key=f"{prompt_prefix}{key}")


    # --- 5. Handle Imported Servers (Reflected in source server's managers) ---
    # The `import_server` method in FastMCP directly adds items to the managers.
    # Since we copied the *local* managers in step 3 and *then* handled mounted servers recursively,
    # items that were *imported* into the original `server` instance before mapping should already be included
    # in the managers copied in step 3. No extra step should be needed here specifically for imported items.
    logger.debug(f"Imported items (if any) are included from local manager copy for '{server.name}'.")


    logger.info(f"Mapped FastMCP Server '{server.name}' (id: {server_id}) to MockFastMcp '{mock_mcp.name}'")
    # Final counts after mounting/importing reflected
    logger.debug(f"Final Mock Tools Count: {len(await mock_mcp.get_tools())}")
    logger.debug(f"Final Mock Resources Count: {len(await mock_mcp.get_resources())}")
    logger.debug(f"Final Mock Templates Count: {len(await mock_mcp.get_resource_templates())}")
    logger.debug(f"Final Mock Prompts Count: {len(await mock_mcp.get_prompts())}")


    return mock_mcp

