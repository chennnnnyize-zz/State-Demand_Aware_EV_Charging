import numpy as np
import csv
import cvxpy as cp
import matplotlib.pyplot as plt


n=3
m=2
T=100

#Online algorithm, do not know SOC
def MPC_Solver(num_of_vehicles, timesteps, initial_states, max_power,
               terminal_states, dept_time, power_capacity, plot_fig):
    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal')
    x0 = cp.Parameter(num_of_vehicles, name='x0')
    max_sum_u = cp.Parameter(name='max_sum_u')
    u_max = cp.Parameter(num_of_vehicles, name='u_max')
    x = cp.Variable((num_of_vehicles, timesteps+1), name='x')
    u = cp.Variable((num_of_vehicles, timesteps), name='u')

    x_terminal.value=terminal_states
    x0.value=initial_states
    max_sum_u.value = power_capacity
    u_max.value=max_power*np.ones((num_of_vehicles, ))

    obj = 0
    constr = [x[:,0] == x0, x[:,-1] <= x_terminal]

    for t in range(timesteps):
        constr += [x[:,t+1] == x[:,t] + u[:,t],
                   u[:,t] <= u_max,
                   u[:,t] >= 0,
                   cp.sum(u[:,t]) <= max_sum_u,
                   u[:,t] <= (t*np.ones_like(dept_time)<dept_time)*100.0+0.000001]
        obj += cp.sum(cp.log(u[:,t]))
    #constr+=[u[5,9]<=0.1]
    obj -= cp.norm(x[:, -1]-x_terminal, 2)
    prob = cp.Problem(cp.Maximize(obj), constr)
    prob.solve()
    #print(x.value[:,-1])
    #print(u.value)

    if plot_fig==True:
        plt.plot(x.value[0])
        plt.plot(u.value[0])
        plt.show()
    #print("DOne")


    return x.value, u.value





#Consider SOC level for the maximum charging power
def MPC_BIG_Solver(num_of_vehicles, timesteps, initial_states,
                   max_power, terminal_states, dept_time,
                   power_capacity, decay_rate, plot_fig):

    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal')
    x0 = cp.Parameter(num_of_vehicles, name='x0')
    max_sum_u = cp.Parameter(name='max_sum_u')
    u_max = cp.Parameter(timesteps, name='u_max')
    x = cp.Variable((num_of_vehicles, timesteps+1), name='x')
    u = cp.Variable((num_of_vehicles, timesteps), name='u')

    x_terminal.value=terminal_states
    x0.value = initial_states
    max_sum_u.value = power_capacity
    u_max.value = max_power * np.ones((timesteps, ))

    obj = 0
    constr = [x[:,0] == x0, x[:,-1] <= x_terminal]

    for i in range(num_of_vehicles):
        constr +=[x[i,1:] == x[i,0:timesteps] + u[i,:],
            u[i,:] <= u_max,
            u[i,:] >= 0,
        ]

    constr += [u <= (max_power - decay_rate * x[:, :-1]),
               x[:, 1:] == x[:, :-1] + u
               ]

    for t in range (timesteps):
        constr += [
                   cp.sum(u[:,t]) <= max_sum_u,
                   u[:,t] <= (t*np.ones_like(dept_time)<dept_time)*100.0+0.000001,
        ]
        obj += cp.sum(cp.log(u[:,t]))
        #obj -= cp.norm(cp.sum(u[:, t]) - max_sum_u, 2)

    #constr+=[u[5,9]<=0.1]
    #constr += [u <= x[:,:-1]]
    obj -= cp.norm(x[:, -1] - x_terminal, 2)
    prob = cp.Problem(cp.Maximize(obj), constr)
    prob.solve()
    #print(x.value[:,-1])
    #print("u value", np.around(u.value,2))
    #print("sum of step-wise power", np.round(np.sum(u.value, axis=0),2))
    #print("x value", np.around(x.value,2))
    #print("DONE")
    if plot_fig==True:
        plt.plot(x.value[0])
        plt.plot(u.value[0])
        plt.show()

    return x.value, u.value



#Consider SOC level for the maximum charging power
#Haven't finished
def MPC_SOC_Solver(num_of_vehicles, timesteps, initial_states, max_power, terminal_states, dept_time, power_capacity):
    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal')
    x0 = cp.Parameter(num_of_vehicles, name='x0')
    max_sum_u = cp.Parameter(name='max_sum_u')
    u_max = cp.Parameter(num_of_vehicles, name='u_max')
    x = cp.Variable((num_of_vehicles, timesteps+1), name='x')
    u = cp.Variable((num_of_vehicles, timesteps), name='u')

    x_terminal.value=terminal_states
    x0.value=initial_states
    max_sum_u.value = power_capacity
    u_max.value=max_power*np.ones((num_of_vehicles, ))

    obj = 0
    print((0 * np.ones_like(dept_time) < dept_time)*10000)
    constr = [x[:,0] == x0, x[:,-1] <= x_terminal]

    for t in range(timesteps):
        constr += [x[:,t+1] == x[:,t] + u[:,t],
                   u[:,t] <= u_max,
                   u[:,t]>=0,
                   cp.sum(u[:,t]) <= max_sum_u,
                   u[:,t] <= ((t*np.ones_like(dept_time) < dept_time)*100.0+0.000001)
                   ]
        obj += cp.sum(cp.log(u[:,t]))
        # A bad grammar error here
        u_max.value=max_power*np.ones((num_of_vehicles, ))#-x[:,t]*0.5
        print("T", t)
        print(u_max.value)
    #constr+=[u[5,9]<=0.1]
    constr += [u<=x[:,:-1]]
    obj -= cp.norm(x[:, -1]-x_terminal, 1)
    prob = cp.Problem(cp.Maximize(obj), constr)
    prob.solve()
    #print(x.value[:,-1])
    #print(u.value)


    #plt.plot(x.value[0])
    #plt.show()

    return x.value, u.value