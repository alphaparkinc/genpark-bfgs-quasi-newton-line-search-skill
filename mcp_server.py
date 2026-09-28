"""MCP stdio server for BFGS Optimization."""
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from client import BFGSSolver

def handle_rpc(request):
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "optimize_quadratic_bowl",
                        "description": "Minimize quadratic objective (x - c1)^2 + (y - c2)^2 via BFGS",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "center": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                "x0": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
                            },
                            "required": ["center", "x0"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "optimize_quadratic_bowl":
            c = args.get("center", [0.0, 0.0])
            x0 = args.get("x0", [0.0, 0.0])
            func = lambda p: (p[0] - c[0])**2 + (p[1] - c[1])**2
            grad = lambda p: [2.0 * (p[0] - c[0]), 2.0 * (p[1] - c[1])]
            res = BFGSSolver.minimize(func, grad, x0)
            return {"jsonrpc": "2.0", "id": req_id, "result": res}
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method {name} not found"}}
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Invalid request"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_rpc(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
