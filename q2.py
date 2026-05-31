

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


# -----------------------------------------------------------
# True function and data sampler
# -----------------------------------------------------------
def f(x):
    t1 = np.sqrt(x * (1 - x))
    t2 = (2.1 * np.pi) / (x + 0.05)
    t3 = np.sin(t2)
    return t1 * t3


def f_sampler(f, n=160, sigma=0.2):
    xvals = np.random.uniform(0, 1, n)
    yvals = f(xvals) + sigma * np.random.normal(0, 1, n)
    return xvals, yvals


# -----------------------------------------------------------
# Gradient-Combination model
# -----------------------------------------------------------
class GCEnsemble:
    def __init__(self):
        self.learners = []
        self.alphas = []

    def predict(self, X):
        if len(self.learners) == 0:
            return np.zeros(X.shape[0])
        total = np.zeros(X.shape[0])
        for h, a in zip(self.learners, self.alphas):
            total += a * h.predict(X)
        return total


def fit_gc_squared_error(X, y, T=50, base_depth=1, step_type="fixed", alpha=0.1):
    """
    Fits Gradient-Combination for squared-error loss.
    step_type: 'fixed' or 'adaptive'
    adaptive alpha_t = (r^T h) / (h^T h)
    """
    model = GCEnsemble()
    f_pred = np.zeros_like(y)

    for t in range(1, T + 1):
        # compute pseudo-residuals
        r = y - f_pred

        # fit base learner to residuals
        tree = DecisionTreeRegressor(max_depth=base_depth)
        tree.fit(X, r)
        h = tree.predict(X)

        # compute step size
        if step_type == "adaptive":
            denom = np.dot(h, h)
            a_t = np.dot(r, h) / denom if denom != 0 else 0.0
        else:
            a_t = alpha

        # update ensemble
        model.learners.append(tree)
        model.alphas.append(a_t)
        f_pred += a_t * h

    return model


# -----------------------------------------------------------
# Plotting helper
# -----------------------------------------------------------
def plot_gc_models(depth=1, step_type="fixed", alpha=0.1, filename="plot.png"):
    np.random.seed(123)
    Xvals, yvals = f_sampler(f, 160, sigma=0.2)
    X, y = Xvals.reshape(-1, 1), yvals

    T_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    fig = plt.figure(figsize=(14, 20))
    xx = np.linspace(0, 1, 1000)

    for idx, T in enumerate(T_list, start=1):
        model = fit_gc_squared_error(X, y, T=T, base_depth=depth,
                                     step_type=step_type, alpha=alpha)
        yy_pred = model.predict(xx.reshape(-1, 1))
        ax = fig.add_subplot(5, 2, idx)
        ax.scatter(X[:, 0], y, marker='x', color='blue', s=10)
        ax.plot(xx, f(xx), color='red', alpha=0.6)
        ax.plot(xx, yy_pred, color='green')
        ax.set_title(f"Depth={depth}, {step_type}, T={T}")
        ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {filename}")


# -----------------------------------------------------------
# Main execution
# -----------------------------------------------------------
if __name__ == "__main__":
    # Generate results for Q2(c) and Q2(d)
    plot_gc_models(depth=1, step_type="fixed",
                   alpha=0.1, filename="Q2c_depth1_fixed_alpha0.1.png")
    plot_gc_models(depth=1, step_type="adaptive",
                   filename="Q2c_depth1_adaptive.png")

    plot_gc_models(depth=2, step_type="fixed",
                   alpha=0.1, filename="Q2d_depth2_fixed_alpha0.1.png")
    plot_gc_models(depth=2, step_type="adaptive",
                   filename="Q2d_depth2_adaptive.png")