import numpy as np

# Simulate trajectory from initial condition x using forward function f: x_t = f(x_{t-1})
def simulate(
    f, # Forward model x_t = f(x_{t-1})
    x, # Initial condition
    n_steps, # Number of steps to simulate forward
):    
    trajectory = []
    # inital condition
    trajectory.append(x)
    
    # Drive model forward and append to new solution
    x_old = np.copy(x)
    for t in range(n_steps-1):
        x_new = f(x_old)
        trajectory.append(x_new)
        x_old = x_new

    return trajectory

def generate_observations(
    x,  # Latent trajectory time series (Nx x T matrix)
    H,  # Observation matrix
    Σy, # Observation noise covariance
):

    # Define observation operator
    def h(x): return np.dot(H, x) + np.random.multivariate_normal( np.zeros(len(Σy)), Σy )

    # generate noisy data
    data = np.array( [h(x_) for x_ in x] )
    return data