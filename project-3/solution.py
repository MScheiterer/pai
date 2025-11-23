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
            alpha=0.15**2,  
            normalize_y=True, 
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
        self.prior_mean_v = 4.0
        self.gp_v = GaussianProcessRegressor(
            kernel=kernel_v,
            alpha=0.0001**2,
            normalize_y=False,  
            n_restarts_optimizer=10
        )
        
        self.kappa = SAFETY_THRESHOLD
        self.f_best = -np.inf
        
        # Lagrangian penalty parameter
        self.lambda_penalty = 40.0  # 40 = 0.83, 50 = 0.815
        
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

        x_opt = self.optimize_acquisition_function()
        return np.array([[x_opt]])

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
        
        sigma_f = np.maximum(sigma_f, 1e-9)
        sigma_v = np.maximum(sigma_v, 1e-9)
        
        # Upper Confidence Bound for objective
        kappa_exploration = 2  
        ucb_f = mu_f + kappa_exploration * sigma_f
        
        # Calculate expected constraint violation using closed form for gaussians
        violation_amount = mu_v - self.kappa
        z = violation_amount / sigma_v
        
        expected_violation = violation_amount * norm.cdf(z) + sigma_v * norm.pdf(z)
        expected_violation = np.maximum(expected_violation, 0)  
        
        # Lagrangian relaxation
        af_value = ucb_f - self.lambda_penalty * expected_violation
        
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
        
        self.gp_f.fit(self.X_f, self.y_f)
        self.gp_v.fit(self.X_v, self.y_v - self.prior_mean_v)
        
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
