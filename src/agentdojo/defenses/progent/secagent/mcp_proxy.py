import argparse
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.proxy import ProxyClient
from .tool import check_tool_call, update_security_policy

server = FastMCP.as_proxy(
    ProxyClient("https://api.githubcopilot.com/mcp/"),
    name="MCP Proxy"
)

policies = {
    'get_me':                       [(1, 0, {}, 0)],
    'search_repositories':          [(1, 0, {}, 0)],
    'list_issues':                  [(1, 0, {
        "repo": {'type': 'string', 'enum': ["pacman"]}
    }, 0)],
    'get_file_contents':            [(1, 0, {
        "repo": {'type': 'string', 'enum': ["pacman"]}
    }, 0)],
    'create_or_update_file':        [(1, 0, {
        "repo": {'type': 'string', 'enum': ["pacman"]}
    }, 0)],
    'create_pull_request':          [(1, 0, {
        "repo": {'type': 'string', 'enum': ["pacman"]}
    }, 0)]
}
update_security_policy(policies)


class ToolCallFilterMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        print(context)
        tool_name, tool_kwargs = context.message.name, context.message.arguments
        check_tool_call(tool_name, tool_kwargs)
        result = await call_next(context)
        return result


server.add_middleware(ToolCallFilterMiddleware())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP Proxy Server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the proxy server on (default: %(default)s)",
    )

    args = parser.parse_args()

    server.run(
        transport="http", host="0.0.0.0", port=args.port, log_level="ERROR", stateless_http=True
    )
