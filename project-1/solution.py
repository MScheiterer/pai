import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    """
    Model for this task.
    You need to implement the train_model and generate_predictions methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        
        self.ensemble = []
        

        kernel_residential = (
            ConstantKernel(1.0, (0.1, 100.0)) *
            Matern(
                length_scale=1.0, 
                length_scale_bounds=(0.01, 5.0), 
                nu=2.5
            ) +
            WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0))
        )

        self.gpr_residential = GaussianProcessRegressor(
                kernel=kernel_residential,
                alpha=1e-2,
                n_restarts_optimizer=2,
                random_state=0,
                normalize_y=True,
                optimizer='fmin_l_bfgs_b',
            )  
        
        kernel_nonresidential = (
            ConstantKernel(1.0, (0.1, 100.0)) *
            Matern(
                length_scale=1.0, 
                length_scale_bounds=(0.01, 5.0), 
                nu=2.5 
            ) +
            WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0))
        )

        self.gpr_nonresidential = GaussianProcessRegressor(
                kernel=kernel_nonresidential,
                alpha=1e-2,
                n_restarts_optimizer=2,
                random_state=0,
                normalize_y=True,
                optimizer='fmin_l_bfgs_b',
            )  
        
        self.adjustment_factor = 1.6
        
    def smart_sample(self, train_coordinates, train_targets, train_area_flags, 
                 n_total=400, residential_ratio=0.7):
        """
        Hybrid: Oversample residential + ensure spatial coverage
        """
        # Split budget between residential and non-residential
        n_residential = int(n_total * residential_ratio)
        n_non_residential = n_total - n_residential
        
        # Get indices for each area type
        residential_idx = np.where(train_area_flags == 1)[0]  # or train_area_flags.astype(bool)
        non_residential_idx = np.where(train_area_flags == 0)[0]  # or ~train_area_flags.astype(bool)
        
        # Use spatial clustering within each group
        from sklearn.cluster import KMeans
        
        def cluster_sample(coords, indices, n):
            if len(indices) <= n:
                return indices
            if len(indices) == 0:  # Handle edge case
                return indices
            kmeans = KMeans(n_clusters=n, random_state=0, n_init=10)
            kmeans.fit(coords[indices])
            selected = []
            for center in kmeans.cluster_centers_:
                distances = np.sum((coords[indices] - center)**2, axis=1)
                selected.append(indices[np.argmin(distances)])
            return np.array(selected)
        
        res_sample = cluster_sample(train_coordinates, residential_idx, n_residential)
        non_res_sample = cluster_sample(train_coordinates, non_residential_idx, n_non_residential)
        
        selected_idx = np.concatenate([res_sample, non_res_sample])
        
        return train_coordinates[selected_idx], train_targets[selected_idx], train_area_flags[selected_idx]

    # Don't change the name or the signature of this function
    def predict_pollution_concentration(self, test_coordinates: np.ndarray, test_area_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_coordinates: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_area_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        n_test = len(test_coordinates)
        gp_mean = np.zeros(n_test)
        gp_std = np.zeros(n_test)
        predictions = np.zeros(n_test)
        
        # Separate test points by area type
        residential_mask = (test_area_flags == 1)
        nonresidential_mask = (test_area_flags == 0)
        
        # Predict residential areas with residential GP
        if np.any(residential_mask):
            X_res_test = test_coordinates[residential_mask]
            mean_res, std_res = self.gpr_residential.predict(X_res_test, return_std=True)
            
            gp_mean[residential_mask] = mean_res
            gp_std[residential_mask] = std_res
            
            # Apply asymmetric cost adjustment for residential
            predictions[residential_mask] = mean_res + self.adjustment_factor * std_res
        
        # Predict non-residential areas with non-residential GP
        if np.any(nonresidential_mask):
            X_nonres_test = test_coordinates[nonresidential_mask]
            mean_nonres, std_nonres = self.gpr_nonresidential.predict(X_nonres_test, return_std=True)
            
            gp_mean[nonresidential_mask] = mean_nonres
            gp_std[nonresidential_mask] = std_nonres
            
            # No adjustment for non-residential (symmetric cost)
            predictions[nonresidential_mask] = mean_nonres
        
        return predictions, gp_mean, gp_std

    # Don't change the name or the signature of this function
    def fit_model_on_training_data(self, train_targets: np.ndarray, train_coordinates: np.ndarray, train_area_flags: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_coordinates: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_targets: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        :param train_area_flags: Binary variable denoting whether the 2D training point is in the residential area (1) or not (0)
        """
        # Split data by area type
        residential_mask = (train_area_flags == 1)
        nonresidential_mask = (train_area_flags == 0)
        
        X_residential = train_coordinates[residential_mask]
        y_residential = train_targets[residential_mask]
        
        X_nonresidential = train_coordinates[nonresidential_mask]
        y_nonresidential = train_targets[nonresidential_mask]
        
        X_residential, y_residential, _ = self.smart_sample(
            X_residential, y_residential, train_area_flags[residential_mask],
            n_total=3000, residential_ratio=1.0
        )
        
        X_nonresidential, y_nonresidential, _ = self.smart_sample(
            X_nonresidential, y_nonresidential, train_area_flags[nonresidential_mask],
            n_total=3000, residential_ratio=0.0
        )
        
        print(f"Residential training points: {len(X_residential)}")
        print(f"Non-residential training points: {len(X_nonresidential)}")
        
        # Train residential GP
        print("\nTraining residential GP...")
        self.gpr_residential.fit(X_residential, y_residential)
        print(f"  Optimized kernel: {self.gpr_residential.kernel_}")
        
        # Train non-residential GP
        print("\nTraining non-residential GP...")
        self.gpr_nonresidential.fit(X_nonresidential, y_nonresidential)
        print(f"  Optimized kernel: {self.gpr_nonresidential.kernel_}")
        
        print("\n=== Training Complete ===")
    

# You don't have to change this function
def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, area_flags: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param area_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(area_flag) for area_flag in area_flags]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def check_within_circle(coordinate, circle_parameters):
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2

# You don't have to change this function 
def identify_city_area_flags(grid_coordinates):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param grid_coordinates: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    area_flags = np.zeros((grid_coordinates.shape[0],))

    for i,coordinate in enumerate(grid_coordinates):
        area_flags[i] = any([check_within_circle(coordinate, circ) for circ in circles])

    return area_flags

# Don't change the name or the signature of this function
def perform_extended_model_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_area_flags = identify_city_area_flags(visualization_grid)
    
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict_pollution_concentration(visualization_grid, grid_area_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()

# Don't change the name or the signature of this function
def get_city_area_data(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_coordinates = np.zeros((train_x.shape[0], 2), dtype=float)
    train_area_flags = np.zeros((train_x.shape[0],), dtype=bool)
    test_coordinates = np.zeros((test_x.shape[0], 2), dtype=float)
    test_area_flags = np.zeros((test_x.shape[0],), dtype=bool)

    #TODO: Extract the city_area information from the training and test features using numpy indexing

    train_coordinates = train_x[:,0:2]
    train_area_flags = train_x[:,2]
    test_coordinates = test_x[:,0:2]
    test_area_flags = test_x[:,2]

    assert train_coordinates.shape[0] == train_area_flags.shape[0] and test_coordinates.shape[0] == test_area_flags.shape[0]
    assert train_coordinates.shape[1] == 2 and test_coordinates.shape[1] == 2
    assert train_area_flags.ndim == 1 and test_area_flags.ndim == 1

    return train_coordinates, train_area_flags, test_coordinates, test_area_flags

# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_coordinates, train_area_flags, test_coordinates, test_area_flags = get_city_area_data(train_x, test_x)
    
    # Fit the model
    print('Training model')
    model = Model()
    model.fit_model_on_training_data(train_y, train_coordinates, train_area_flags)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.predict_pollution_concentration(test_coordinates, test_area_flags)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_model_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
