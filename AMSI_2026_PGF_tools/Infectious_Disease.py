import numpy as np

def Gillespie_SIR_model(
    S_IC,
    I_IC,
    R_IC,
    beta,
    gamma,
    tmax=float("inf"),
    rng=None
):
    """
    Simulate the SIR model using the Gillespie algorithm.

    Arguments:
    S_IC -- initial susceptible population
    I_IC -- initial infected population
    R_IC -- initial recovered population (not R_0)
    beta -- transmission rate
    gamma -- recovery rate
    tmax -- maximum time to simulate.
    rng -- optional random number generator. If None, a default
           random number generator will be used.


    Returns:
    t -- np array
            time points
    S -- np array
            susceptible population over time
    I -- np array
            infected population over time
    R -- np array
            recovered population over time

    """

    tstart = 0
    if rng is None:
        rng = np.random.default_rng()

    # Initialize populations and time
    N = S_IC + I_IC + R_IC
    S = [S_IC]
    I = [I_IC]
    R = [R_IC]
    t = [tstart]

    while t[-1] < tmax and I[-1] > 0:
        rec_rate = gamma * I[-1]
        inf_rate = beta * S[-1] * I[-1] / N
        total_rate = rec_rate + inf_rate

        delay = rng.exponential(1 / total_rate)

        time = t[-1] + delay
        if time > tmax:
            t.append(tmax)
            S.append(S[-1])
            I.append(I[-1])
            R.append(R[-1])
            # If there are still infected individuals at tmax, we truncate the simulation and record their data.
            break
        else:
            t.append(time)
            if rng.random() < inf_rate / total_rate:
                # Infection event
                S.append(S[-1] - 1)
                I.append(I[-1] + 1)
                R.append(R[-1])
            else:
                # Recovery event
                S.append(S[-1])
                I.append(I[-1] - 1)
                R.append(R[-1] + 1)

    return np.array(t), np.array(S), np.array(I), np.array(R)

def Gillespie_SIS_model(
    S_IC,
    I_IC,
    beta,
    gamma,
    tmax=float(10),
    rng=None,
    return_cumulative = False,
    cum_max = None
):
    
    """
    Simulate the SIS model using the Gillespie algorithm.

    Arguments:
    S_IC -- initial susceptible population
    I_IC -- initial infected population
    beta -- transmission rate
    gamma -- recovery rate
    tmax -- maximum time to simulate.
    rng -- optional random number generator. If None, a default
           random number generator will be used.
    return_cumulative -- if True, also return cumulative infections over time
    cum_max -- if provided, maximum cumulative infections to record (to save memory)
               Only works if return_cumulative is True.

    Returns:
    t -- np array
            time points
    S -- np array
            susceptible population over time
    I -- np array
            infected population over time
    """

    tstart = 0
    if rng is None:
        rng = np.random.default_rng()

    if cum_max is not None and not return_cumulative:
        raise ValueError("cum_max can only be used if return_cumulative is True.")
    # Initialize populations and time
    N = S_IC + I_IC 
    S = [S_IC]
    I = [I_IC]
    t = [tstart]
    if return_cumulative:
        cumulative_infections = [I_IC]

    while t[-1] < tmax and I[-1] > 0 and (return_cumulative==False or (cum_max is None or cumulative_infections[-1]<cum_max)):
        rec_rate = gamma * I[-1]
        inf_rate = beta * S[-1] * I[-1] / N
        total_rate = rec_rate + inf_rate

        delay = rng.exponential(1 / total_rate)

        time = t[-1] + delay
        if time > tmax:
            t.append(tmax)
            S.append(S[-1])
            I.append(I[-1])
            if return_cumulative:
                cumulative_infections.append(cumulative_infections[-1])
            # If there are still infected individuals at tmax, we truncate the simulation and record their data.
            break
        else:
            t.append(time)
            if rng.random() < inf_rate / total_rate:
                # Infection event
                S.append(S[-1] - 1)
                I.append(I[-1] + 1)
                if return_cumulative:
                    cumulative_infections.append(cumulative_infections[-1] + 1)
            else:
                # Recovery event
                S.append(S[-1]+1)
                I.append(I[-1] - 1)
                if return_cumulative:
                    cumulative_infections.append(cumulative_infections[-1])
    if return_cumulative:
        return np.array(t), np.array(S), np.array(I), np.array(cumulative_infections)
    return np.array(t), np.array(S), np.array(I)

