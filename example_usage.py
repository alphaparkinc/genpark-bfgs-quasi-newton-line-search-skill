"""Example demonstrating BFGS optimization."""
from client import BFGSSolver

def main():
    f = lambda p: (p[0] - 3.0)**2 + (p[1] + 2.0)**2
    g = lambda p: [2.0 * (p[0] - 3.0), 2.0 * (p[1] + 2.0)]
    res = BFGSSolver.minimize(f, g, x0=[0.0, 0.0])
    print("BFGS Optimization Result:")
    print("  Optimal Solution:", res["x"])
    print("  Objective Value:", res["f_val"])
    print("  Iterations:", res["iterations"])

if __name__ == "__main__":
    main()
