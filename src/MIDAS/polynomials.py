import numpy as np

def polynomial_weights(name):
    weights_classes = {
        'beta': BetaWeights,
        'beta_nz': BetaWeights,  # same implementation
        'expalmon': ExpAlmonWeights
    }

    if name not in weights_classes:
        raise ValueError(f"Unknown polynomial weight method: {name}")

    return weights_classes[name]()  # instantiate with default params


class WeightMethod:
    def __init__(self):
        self.theta = self.init_params()

    def weights(self, nlags):
        raise NotImplementedError

    def x_weighted(self, x, params):
        self.theta = np.array(params)
        w = self.weights(x.shape[1])
        return np.dot(x, w), np.tile(w.T, (x.shape[1], 1))

    def init_params(self):
        raise NotImplementedError

    @property
    def num_params(self):
        return len(self.init_params())


class BetaWeights(WeightMethod):
    def weights(self, nlags):
        eps = np.spacing(1)
        u = np.linspace(eps, 1.0 - eps, nlags)
        theta1, theta2 = self.theta[:2]
        beta_vals = u ** (theta1 - 1) * (1 - u) ** (theta2 - 1)
        beta_vals = beta_vals / sum(beta_vals)

        if len(self.theta) == 3:
            beta_vals += self.theta[2]
            beta_vals = beta_vals / sum(beta_vals)

        return beta_vals

    def init_params(self):
        return np.array([1.0, 5.0])  # you can extend to [1.0, 5.0, 0.0] for beta_nz if needed


class ExpAlmonWeights(WeightMethod):
    def weights(self, nlags):
        ilag = np.arange(1, nlags + 1)
        theta1, theta2 = self.theta
        z = np.exp(theta1 * ilag + theta2 * ilag ** 2)
        return z / np.sum(z)

    def init_params(self):
        return np.array([-1.0, 0.0])
