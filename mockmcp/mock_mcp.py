from __future__ import annotations as _annotations

import contextvars
from typing import Any, TypeVar, Callable, Awaitable, Generic, Iterable
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, AbstractAsyncContextManager
import warnings
import logging

import mcp.types as types
from mcp.server.lowlevel.server import Server, request_ctx, _ping_handler, LifespanResultT
from mcp.shared.context import RequestContext
from mcp.shared.session import BaseSession, RequestResponder
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl
from mcp.server.lowlevel.server import lifespan

logger = logging.getLogger(__name__)

# A placeholder Mock Session
class MockSession(BaseSession):
    """A mock session that does nothing, for use with MockMcp."""
    async def send_request(self, request, result_type):
        raise NotImplementedError("MockSession cannot send requests")

    async def send_notification(self, notification):
        logger.debug(f"MockSession received notification (ignored): {notification}")
        # In a mock environment, we might just log or ignore notifications
        pass

    async def _handle_incoming(self, req):
        pass # No incoming messages to handle in the mock


@asynccontextmanager
async def mock_lifespan(server: MockMcp[LifespanResultT]) -> AsyncIterator[object]:
    """Default lifespan context manager for MockMcp that does nothing."""
    yield {}


class MockMcp(Generic[LifespanResultT]):
    """
    A mock MCP server implementation that allows direct invocation of handlers
    without network communication.
    """
    def __init__(
        self,
        name: str = "mock_server",
        version: str | None = None,
        lifespan_context: LifespanResultT | None = None,
    ):
        self.name = name
        self.version = version
        self.request_handlers: dict[
            type, Callable[..., Awaitable[types.ServerResult]]
        ] = {
            types.PingRequest: _ping_handler, # Keep ping handler for basic check
        }
        self.notification_handlers: dict[type, Callable[..., Awaitable[None]]] = {}
        self._lifespan_context = lifespan_context if lifespan_context is not None else {} # type: ignore
        self._mock_session = MockSession(None, None, types.ClientRequest, types.ClientNotification) # type: ignore

    def _register_request_handler(self, request_type: type, handler: Callable[..., Awaitable[types.ServerResult]]):
        logger.debug(f"MockRegistering handler for {request_type.__name__}")
        self.request_handlers[request_type] = handler

    def _register_notification_handler(self, notification_type: type, handler: Callable[..., Awaitable[None]]):
         logger.debug(f"MockRegistering notification handler for {notification_type.__name__}")
         self.notification_handlers[notification_type] = handler

    async def _call_handler(self, request: types.ServerRequest | Any) -> Any:
        """
        Internal helper to simulate calling a request handler.
        Sets up mock context and extracts the actual result.
        """
        req_type = type(request)
        if req_type not in self.request_handlers:
            raise McpError(types.ErrorData(code=types.METHOD_NOT_FOUND, message=f"Method '{request.method}' not found in mock server")) # type: ignore

        handler = self.request_handlers[req_type]
        token = None
        response = None
        try:
            # Set up a mock context
            mock_request_id = "mock_req_0"
            mock_meta = getattr(request.params, '_meta', None) if getattr(request, 'params', None) else None

            ctx = RequestContext(
                request_id=mock_request_id,
                meta=mock_meta,
                session=self._mock_session, # Use the mock session
                lifespan_context=self._lifespan_context,
            )
            token = request_ctx.set(ctx)

            # Call the handler (which expects the request object and returns ServerResult)
            server_result = await handler(request)

            # Extract the actual result from ServerResult
            if isinstance(server_result, types.ServerResult) and hasattr(server_result.root, 'model_dump'):
                 # Extract the inner result (e.g., ListPromptsResult, CallToolResult)
                 response = server_result.root
            elif isinstance(server_result, types.ServerResult) and server_result.root is None:
                 # Handle cases like Ping where the result is EmptyResult leading to root=None
                 response = None
            else:
                 # Should not happen if handlers adhere to Server class structure
                 logger.warning(f"Handler for {req_type.__name__} returned unexpected type: {type(server_result)}")
                 response = server_result # Or raise error

        except McpError as err:
            # Re-raise MCP errors directly for clarity in mocking
            raise err
        except Exception as err:
            logger.error(f"Error calling mock handler for {req_type.__name__}: {err}", exc_info=True)
            # Simulate error response generation if needed, or just raise
            raise McpError(types.ErrorData(code=types.INTERNAL_ERROR, message=str(err)))
        finally:
            if token:
                request_ctx.reset(token)

        # Further extract specific fields if needed (e.g., from CallToolResult)
        if isinstance(response, types.CallToolResult):
             if response.isError:
                  # Try to extract the error message if available
                  error_text = "Tool execution failed"
                  if response.content and isinstance(response.content[0], types.TextContent):
                       error_text = response.content[0].text
                  raise McpError(types.ErrorData(code=1, message=error_text)) # Using custom code 1 for tool error
             return response.content # Return the actual content list
        elif isinstance(response, types.GetPromptResult):
            return response.messages # Return the messages list
        elif isinstance(response, types.ListPromptsResult):
            return response.prompts
        elif isinstance(response, types.ListToolsResult):
            return response.tools
        elif isinstance(response, types.ListResourcesResult):
            return response.resources
        elif isinstance(response, types.ListResourceTemplatesResult):
            return response.resourceTemplates
        elif isinstance(response, types.ReadResourceResult):
            # The original handler might return ServerResult(ReadResourceResult(...))
            # We need the contents list.
            return response.contents
        elif isinstance(response, types.CompleteResult):
            return response.completion
        elif response is None and req_type is types.PingRequest:
             return None # Ping returns EmptyResult -> None
        elif response is None and (req_type is types.SetLevelRequest or req_type is types.SubscribeRequest or req_type is types.UnsubscribeRequest):
             return None # These also return EmptyResult -> None

        # Fallback or handle other result types
        return response

    # --- Direct Call Methods ---

    async def list_prompts(self) -> list[types.Prompt]:
        request = types.ListPromptsRequest(method="prompts/list", params=None)
        return await self._call_handler(request) # type: ignore

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> list[types.PromptMessage]:
        params = types.GetPromptRequestParams(name=name, arguments=arguments)
        request = types.GetPromptRequest(method="prompts/get", params=params)
        result: list[types.PromptMessage] = await self._call_handler(request)
        return result

    async def list_tools(self) -> list[types.Tool]:
        request = types.ListToolsRequest(method="tools/list", params=None)
        return await self._call_handler(request) # type: ignore

    async def call_tool(
        self, name: str, arguments: dict | None = None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        params = types.CallToolRequestParams(name=name, arguments=arguments or {})
        request = types.CallToolRequest(method="tools/call", params=params)
        result: list[types.TextContent | types.ImageContent | types.EmbeddedResource] = await self._call_handler(request)
        return result

    async def list_resources(self) -> list[types.Resource]:
         request = types.ListResourcesRequest(method="resources/list", params=None)
         return await self._call_handler(request) # type: ignore

    async def list_resource_templates(self) -> list[types.ResourceTemplate]:
         request = types.ListResourceTemplatesRequest(method="resources/templates/list", params=None)
         return await self._call_handler(request) # type: ignore

    async def read_resource(self, uri: AnyUrl) -> list[types.TextResourceContents | types.BlobResourceContents]:
        """Reads a resource and returns the list of contents directly."""
        params = types.ReadResourceRequestParams(uri=uri)
        request = types.ReadResourceRequest(method="resources/read", params=params)
        result: list[types.TextResourceContents | types.BlobResourceContents] = await self._call_handler(request)
        return result

    async def set_logging_level(self, level: types.LoggingLevel) -> None:
        params = types.SetLevelRequestParams(level=level)
        request = types.SetLevelRequest(method="logging/setLevel", params=params)
        await self._call_handler(request)

    async def subscribe_resource(self, uri: AnyUrl) -> None:
        params = types.SubscribeRequestParams(uri=uri)
        request = types.SubscribeRequest(method="resources/subscribe", params=params)
        await self._call_handler(request)

    async def unsubscribe_resource(self, uri: AnyUrl) -> None:
        params = types.UnsubscribeRequestParams(uri=uri)
        request = types.UnsubscribeRequest(method="resources/unsubscribe", params=params)
        await self._call_handler(request)

    async def complete(
        self,
        ref: types.PromptReference | types.ResourceReference,
        argument: types.CompletionArgument,
    ) -> types.Completion:
        params = types.CompleteRequestParams(ref=ref, argument=argument)
        request = types.CompleteRequest(method="completion/complete", params=params)
        result: types.Completion = await self._call_handler(request)
        return result

    async def ping(self) -> None:
        request = types.PingRequest(method="ping", params=None)
        await self._call_handler(request)

    # --- Decorators (for potential direct use with MockMcp) ---
    # These mimic Server's decorators but register handlers directly on MockMcp

    def list_prompts_decorator(self):
        def decorator(func: Callable[[], Awaitable[list[types.Prompt]]]):
            async def handler(_: Any):
                prompts = await func()
                return types.ServerResult(types.ListPromptsResult(prompts=prompts))
            self._register_request_handler(types.ListPromptsRequest, handler)
            return func
        return decorator

    def get_prompt_decorator(self):
        def decorator(
            func: Callable[[str, dict[str, str] | None], Awaitable[types.GetPromptResult]]
        ):
            async def handler(req: types.GetPromptRequest):
                prompt_get = await func(req.params.name, req.params.arguments)
                return types.ServerResult(prompt_get)
            self._register_request_handler(types.GetPromptRequest, handler)
            return func
        return decorator

    # ... Add other decorators similarly if needed ...
    # Example for call_tool decorator:
    def call_tool_decorator(self):
        def decorator(
            func: Callable[
                ..., Awaitable[Iterable[types.TextContent | types.ImageContent | types.EmbeddedResource]]
            ],
        ):
            async def handler(req: types.CallToolRequest):
                try:
                    # Note: Server wraps the call directly. We mimic that.
                    results = await func(req.params.name, (req.params.arguments or {}))
                    return types.ServerResult(
                        types.CallToolResult(content=list(results), isError=False)
                    )
                except Exception as e:
                    # Mimic Server's exception handling for tool calls
                    return types.ServerResult(
                        types.CallToolResult(
                            content=[types.TextContent(type="text", text=str(e))],
                            isError=True,
                        )
                    )
            self._register_request_handler(types.CallToolRequest, handler)
            return func
        return decorator


# --- Mapping Function ---

async def map_server_to_mock(server: Server[LifespanResultT]) -> MockMcp[LifespanResultT]:
    """
    Creates a MockMcp instance from an existing Server instance,
    copying its handlers and lifespan context.
    """
    # Run the server's lifespan to get the context, if it has one other than the default
    lifespan_context: LifespanResultT | object
    if server.lifespan == lifespan: # Check if it's the default do-nothing lifespan
         lifespan_context = {}
    else:
         # Execute the server's lifespan to get the context object
         # We need an async context manager to enter the lifespan
         @asynccontextmanager
         async def get_lifespan_ctx():
              async with server.lifespan(server) as ctx:
                   yield ctx
         async with get_lifespan_ctx() as ctx:
              lifespan_context = ctx


    mock_mcp = MockMcp[LifespanResultT](
        name=f"mock_{server.name}",
        version=server.version,
        lifespan_context=lifespan_context # type: ignore
    )

    # Copy handlers directly - they are already wrapped by the Server decorators
    mock_mcp.request_handlers.update(server.request_handlers)
    mock_mcp.notification_handlers.update(server.notification_handlers)

    logger.info(f"Mapped Server '{server.name}' to MockMcp '{mock_mcp.name}'")
    logger.debug(f"Mock Request Handlers: {list(mock_mcp.request_handlers.keys())}")
    logger.debug(f"Mock Notification Handlers: {list(mock_mcp.notification_handlers.keys())}")

    return mock_mcp 