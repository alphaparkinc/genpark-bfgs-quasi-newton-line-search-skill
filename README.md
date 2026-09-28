# BFGS Quasi-Newton Optimizer Skill

Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton optimization featuring rank-2 inverse Hessian approximation updates.

```mermaid
flowchart TD
    Grad["Compute Current Gradient g_k"] --> Direction["Search Direction: p_k = -H_k * g_k"]
    Direction --> LineSearch["Armijo Backtracking Line Search for Step α"]
    LineSearch --> Update["Update State: x_{k+1} = x_k + α * p_k"]
    Update --> Hessian["Rank-2 Inverse Hessian Update H_{k+1}"]
    Hessian --> Conv{"Norm(g) < Tol?"}
    Conv -- No --> Grad
    Conv -- Yes --> Done["Optimal Solution Found"]
```

## Features
- **100% Python Standard Library**: No numpy or scipy required.
- **Superlinear Convergence**: Approximates Newton-Raphson curvature without computing explicit second derivatives.
- **Robust Armijo Line Search**: Guaranteed decrease in objective function.
