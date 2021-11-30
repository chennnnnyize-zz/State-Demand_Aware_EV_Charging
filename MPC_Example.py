import numpy as np
import csv
import cvxpy as cp
#import matplotlib.pyplot as plt


num_of_vehicles=3
timestep=5
T=10

'''A = cp.Parameter((n, n), name='A')
A.value=[[1,0.1,0.1],[0.1,1,0.1],[0.1,0.1,1]]
B = cp.Parameter((n, m), name='B')
B.value=np.array([[1,0],[0,1],[1,1]])
x0 = cp.Parameter(n, name='x0')
x0.value=np.array([0.5,0.5,0.5])
u_max = cp.Parameter(name='u_max')
u_max.value=1.0
x = cp.Variable((n, T+1), name='x')
u = cp.Variable((m, T), name='u')
obj = 0
constr = [x[:,0] == x0, x[:,-1] == 0]
for t in range(T):
    constr += [x[:,t+1] == A@x[:,t] + B@u[:,t], 
    cp.norm(u[:,t], 'inf') <= u_max]
    obj += cp.sum_squares(x[:,t])
prob = cp.Problem(cp.Minimize(obj), constr)
prob.solve()
print(x.value)'''

x = cp.Variable((num_of_vehicles, timestep+1), name='x')
u = cp.Variable((num_of_vehicles, timestep), name='u')
#x0 = cp.Parameter((1,1), name='x0')
#x0.value=0.1
obj = 0
constr = [x[0,0] == 0.1]
constr += [x[1,0] == 0.1]
constr += [x[2,0] == 0.1]
constr += [u <= (0.5 - 0.1*x[:,:-1]),
           x[:,1:]==x[:,:-1]+u
           ]
obj += cp.sum(cp.log(u[:, :]))
prob = cp.Problem(cp.Maximize(obj), constr)
prob.solve()
print(x.value)
print(u.value)
