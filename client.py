"""BFGS Quasi-Newton Optimization Algorithm.
100% Python Standard Library.
"""

import math

def dot(u, v):
    return sum(a * b for a, b in zip(u, v))

class BFGSSolver:
    """Quasi-Newton unconstrained minimizer with inverse Hessian approximation."""
    @staticmethod
    def minimize(func, grad_func, x0, tol=1e-5, max_iter=50):
        n = len(x0)
        x = list(x0)
        H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        
        for k in range(max_iter):
            g = grad_func(x)
            g_norm = math.sqrt(sum(gi**2 for gi in g))
            if g_norm < tol:
                break
                
            p = [-sum(H[i][j] * g[j] for j in range(n)) for i in range(n)]
            
            alpha = 1.0
            c1 = 1e-4
            fx = func(x)
            slope = dot(g, p)
            while func([x[i] + alpha * p[i] for i in range(n)]) > fx + c1 * alpha * slope:
                alpha *= 0.5
                if alpha < 1e-8:
                    break
                    
            x_next = [x[i] + alpha * p[i] for i in range(n)]
            s = [x_next[i] - x[i] for i in range(n)]
            g_next = grad_func(x_next)
            y = [g_next[i] - g[i] for i in range(n)]
            
            ys = dot(y, s)
            if ys > 1e-8:
                rho = 1.0 / ys
                V = [[(1.0 if i == j else 0.0) - rho * y[i] * s[j] for j in range(n)] for i in range(n)]
                HV = [[sum(H[i][l] * V[l][j] for l in range(n)) for j in range(n)] for i in range(n)]
                V_T = [[V[j][i] for j in range(n)] for i in range(n)]
                H_next = [[sum(V_T[i][l] * HV[l][j] for l in range(n)) + rho * s[i] * s[j] for j in range(n)] for i in range(n)]
                H = H_next
                
            x = x_next
            
        return {"x": [round(xi, 5) for xi in x], "f_val": round(func(x), 6), "iterations": k + 1}
