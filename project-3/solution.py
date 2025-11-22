"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import norm
from sklearn.gaussian_process.kernels import DotProduct


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.        
        # Store observed data
        self.X_f = np.array([]).reshape(0, 1)
        self.y_f = np.array([])
        self.X_v = np.array([]).reshape(0, 1)
        self.y_v = np.array([])
        
        kernel_f = C(0.5, (1e-3, 1e3)) * Matern(
            length_scale=1.0, 
            nu=2.5, 
            length_scale_bounds=(0.1, 10.0)
        )
        self.gp_f = GaussianProcessRegressor(
            kernel=kernel_f, 
            alpha=0.15**2,  # noise variance σ_f² = 0.15²
            normalize_y=True,  # helps with numerical stability
            n_restarts_optimizer=10
        )
        
        linear_kernel = DotProduct(sigma_0=0.0, sigma_0_bounds="fixed")
        matern_kernel = Matern(
            length_scale=1.0,
            nu=2.5,
            length_scale_bounds=(0.1, 10.0)
        )
        kernel_v = C(np.sqrt(2), (1e-3, 1e3)) * (linear_kernel + matern_kernel)
        
        # Create GP with prior mean = 4
        # sklearn doesn't directly support non-zero mean, so we shift data
        self.prior_mean_v = 4.0
        self.gp_v = GaussianProcessRegressor(
            kernel=kernel_v,
            alpha=0.0001**2,  # noise variance σ_v² = 0.0001²
            normalize_y=False,  # Don't normalize since we handle mean manually
            n_restarts_optimizer=10
        )
        
        self.kappa = SAFETY_THRESHOLD
        self.f_best = -np.inf
        self.beta = 2.0  # Safety margin (conservative)

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        # Optimize acquisition function
        x_opt = self.optimize_acquisition_function()
        x_opt_2d = np.array([[x_opt]])

        # Predict constraint
        mu_v, sigma_v = self.gp_v.predict(x_opt_2d, return_std=True)
        mu_v = mu_v[0] + self.prior_mean_v
        
        # If upper confidence bound suggests unsafe, be more conservative
        if mu_v + self.beta * sigma_v > self.kappa:
            # Sample many points and find safe ones with good objective
            x_sample = np.random.uniform(DOMAIN[0, 0], DOMAIN[0, 1], (200, 1))
            mu_v_sample, sigma_v_sample = self.gp_v.predict(x_sample, return_std=True)
            mu_v_sample = mu_v_sample + self.prior_mean_v

            # Find points likely to be safe
            safety_margin = mu_v_sample + self.beta * sigma_v_sample #works better with margin same as the check condition
            safe_mask = safety_margin <= self.kappa

            if np.any(safe_mask):
                # Choose point with best UCB objective among safe points
                mu_f_sample, sigma_f_sample = self.gp_f.predict(x_sample[safe_mask], return_std=True)
                gamma = 1.0  # UCB exploration parameter
                ucb_values = mu_f_sample + gamma * sigma_f_sample
                best_idx = np.argmax(ucb_values)
                x_opt_2d = x_sample[safe_mask][best_idx:best_idx + 1]
            else:
                # fallback: safest point
                min_idx = np.argmin(safety_margin)
                x_opt_2d = x_sample[min_idx:min_idx + 1]

        return x_opt_2d

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.       
        mu_f, sigma_f = self.gp_f.predict(x, return_std=True)
        mu_v, sigma_v = self.gp_v.predict(x, return_std=True)
        mu_v = mu_v + self.prior_mean_v  # Add back prior mean
        
        # Avoid division by zero
        sigma_f = np.maximum(sigma_f, 1e-9)
        sigma_v = np.maximum(sigma_v, 1e-9)
        
        # Expected Improvement for objective
        if self.f_best == -np.inf:
            ei = mu_f
        else:
            z = (mu_f - self.f_best) / sigma_f
            ei = (mu_f - self.f_best) * norm.cdf(z) + sigma_f * norm.pdf(z)
        
        # Conservative Probability of Feasibility with safety margin
        # Use mu_v + beta*sigma_v <= kappa (more conservative)
        safe_threshold = self.kappa - self.beta * sigma_v
        pof = norm.cdf((safe_threshold - mu_v) / sigma_v)
        
        # Penalize points with high uncertainty in constraint
        # This discourages exploration in uncertain regions
        uncertainty_penalty = np.exp(-sigma_v / 2.0)
        
        # Combined acquisition with stronger safety emphasis
        af_value = ei * (pof ** 2) * uncertainty_penalty  # Square PoF for more conservative behavior
        
        # Return scalar for single point
        if x.shape[0] == 1:
            return float(af_value[0])
        else:
            return af_value.reshape(-1, 1)

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float or np.ndarray
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        # Ensure x is always 2D array with shape (1, n_features)
        x = np.array(x).reshape(1, -1)
        
        # Add to dataset
        if self.X_f.size == 0:
            self.X_f = x
        else:
            self.X_f = np.vstack([self.X_f, x])
        
        self.y_f = np.append(self.y_f, f)
        
        if self.X_v.size == 0:
            self.X_v = x
        else:
            self.X_v = np.vstack([self.X_v, x])
        
        self.y_v = np.append(self.y_v, v)
        
        # Fit GPs
        self.gp_f.fit(self.X_f, self.y_f)
        self.gp_v.fit(self.X_v, self.y_v - self.prior_mean_v)
        
        # Update best safe objective value
        safe_indices = self.y_v <= self.kappa
        if np.any(safe_indices):
            self.f_best = np.max(self.y_f[safe_indices])
        
    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        # Find all safe points
        safe_indices = self.y_v <= self.kappa
        
        if not np.any(safe_indices):
            # Fallback: return middle of domain
            return np.array([[DOMAIN[0, 0] + 0.5 * (DOMAIN[0, 1] - DOMAIN[0, 0])]])
        
        # Return the safe point with highest objective value
        safe_X = self.X_f[safe_indices]
        safe_y = self.y_f[safe_indices]
        best_idx = np.argmax(safe_y)
        
        return safe_X[best_idx].reshape(1, -1)

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
