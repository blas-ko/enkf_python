import numpy as np

def estimate(
    f, # Forward model. It should be a function that gives the state of the system at next timestep: x_t = f(x_{t-1})
    data, # Ny x T array of empirical observations
    particles, # Ne x Nx array of particles
    H, # Observation operator. Here, an Ny x Nx matrix
    Σy, # Ny x Ny observations covariance
    timestamps=None, # time indices at which observations are sampled # TODO.
    keep_ensemble_history=False, # TODO
    **kw,
):

    ## PREALLOCATION ## 
    filtered_states_analysis = []
    filtered_observations = []

    # TODO: Manage timestamps being a number representing the sampling rate
    if timestamps is not None:
        Δtimestamps = [-1] + list(timestamps)
        Δtimestamps = np.diff(Δtimestamps)
    
    Xa = np.copy(particles)
    for t, y in enumerate(data):
        
        # Update forecast # TODO: Irregular timestamps
        if timestamps is not None:
            # Evolve system for the given number of irregular steps
            Δt = Δtimestamps[t]
            for step in range(Δt):
                Xa = np.array( [f(x, **kw) for x in Xa] )
            Xf = np.copy(Xa)
        else:
            Xf = np.array( [f(x, **kw) for x in Xa] )
            
        # Analysis step using ensemble Kalman filter
        Xa, xa = analysis(H, Xf, y, Σy )

        # Append results
        filtered_states_analysis.append( xa )
        filtered_observations.append( np.dot(H, xa) )
    
        if keep_ensemble_history:
            pass # TODO
            
    # Handle output results
    results = ( np.array(filtered_observations), np.array(filtered_states_analysis)) #, np.array(filtered_states_forecast) )
    if keep_ensemble_history:
        pass
        # results = (results..., Xa_hist, Γ_hist)

    return results

def analysis(
    H, # observation operator
    particles, # ensemble of state particles
    y, # observation
    Σy, # observation covariance (also called R in literature)    
):

    # Process particles and observation (add noise to observation, as in Burgers et al.)
    X = particles
    x_mean, Xp, Yp, Y = process_analysis_elements(X, y, Σy)    

    # particle observations: compute observation operator on each particle
    S = np.array( [np.dot(H,x) for x in X] )

    # sample covariances: state, observations 
    P = np.dot( Xp.T, Xp ) # Same as Σx but empirica.
    R = np.dot( Yp.T, Yp ) # Same as Σy but empirical

    # innovation
    D = Y - S

    # Analysis step:
    aux1 = np.dot(P, H.T) # not useful anymore    # P x H.T
    aux2 = np.linalg.inv( np.dot(H, aux1) + R)    # S^-1:= (H x P x H.T + R)^-1
    aux3 = np.dot(aux1, aux2)                     # K:= (P x H.T x S^-1)
    # Kalman gain × innovation
    KD = np.dot( aux3, D.T ).T                    # K x D

    # Compute analysis for every particle from Kalman gain
    Xa = X + KD
    # Get estimation as average analysis particle
    xa = np.mean(Xa, axis=0)
    
    return Xa, xa

# Helper: creates ensemble anomalies and perturbed datapoints.
def process_analysis_elements(
    particles,
    datapoint, # Ny-sized vector representing observation at current timestep
    Σy, # Observation covariance
):

    ## Forecast Ensemble
    # Ensemble mean
    particles_mean = np.mean(particles, axis=0) # Nₓ-vector
    # Ensemble perturbations
    particles_anomaly = particles - particles_mean # Nₓ × Nₑ 

    # model dimensions, n_particles, observations dimension
    Ne, Ny = np.shape(particles)
    Ny = np.size(datapoint)

    ## Observation perturbations
    datapoint_perturbations = np.random.multivariate_normal( np.zeros(Ny), Σy, size=(Ny,Ne) )[0]
    # Perturbed observations
    datapoint_perturbed = datapoint_perturbations + datapoint # N_y × N_e
    
    return particles_mean, particles_anomaly, datapoint_perturbations, datapoint_perturbed