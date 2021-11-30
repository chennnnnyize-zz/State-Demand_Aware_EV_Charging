import numpy as np
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt


def charging_session(n): #Simulate Charging session from distribution
    norm_params = np.array([[5, 1],
                            [1, 1.3],
                            [9, 1.3]])
    n_components = norm_params.shape[0]
    # Weight of each component, in this case all of them are 1/3
    weights = np.ones(n_components, dtype=np.float64) / 3.0
    # A stream of indices from which to choose the component
    mixture_idx = numpy.random.choice(len(weights), size=n, replace=True, p=weights)
    print(np.shape(mixture_idx))
    # y is the mixture sample
    y = numpy.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                       dtype=np.float64)

    # Theoretical PDF plotting -- generate the x and y plotting positions
    xs = np.linspace(y.min(), y.max(), 200)
    ys = np.zeros_like(xs)
    print(np.shape(ys))

    for (l, s), w in zip(norm_params, weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w

    plt.plot(xs, ys)
    plt.hist(y, normed=True, bins="fd")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    arr_time=ys
    dep_time=xs

    return arr_time, dep_time

def charging_session_individual(lambda_val, vehicle_capacity): #Simulate Charging session from distribution
    arr_time=(np.random.poisson(lam=lambda_val)+5.0)*12.0+np.random.randint(0.0, 12.0)
    sess_time=np.random.randint(1.0, 15.0)*12.0+np.random.randint(0.0, 12.0)
    init_cap=np.random.uniform(0.0, vehicle_capacity-1.0)
    req_energy=np.random.uniform(init_cap+0.5, vehicle_capacity)
    return arr_time, sess_time, init_cap, req_energy