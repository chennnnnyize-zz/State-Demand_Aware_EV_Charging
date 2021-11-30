import numpy as np
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt
from offline_solver import MPC_BIG_Solver, MPC_Solver
import csv
import pandas as pd

#One day has 24*12=288 slots

n = 10000
np.random.seed(0x5ee2)
num_steps=96
decay_rate=0.03
max_power=0.6
total_vehicles=200
battery_capacity=8.0
power_capacity=14.0




def check_SOC_update(initial_states, decay_rate, charging_rate, max_power):
    #print("Initial states", initial_states)
    updated_state=np.round(initial_states + np.minimum(charging_rate[:,0], max_power-decay_rate*initial_states), 2)
    #print(updated_state)
    return updated_state

arrival_time=np.random.poisson(lam=9.0, size=(total_vehicles,))
arrival_time = np.sort(arrival_time)*4.0
arrival_time =arrival_time+np.random.randint(0,4, size=(total_vehicles,))
arrival_time = np.sort(arrival_time) #Session is defined based on arrival time
#plt.hist(arrival_time, normed=True, bins=10)
#plt.ylabel("f(x)")
#plt.show()
depart_time = np.random.randint(6, 36, size=(total_vehicles,))
depart_time = np.min((arrival_time+depart_time, np.ones_like(arrival_time)*96), axis=0)

print("Arrival time", arrival_time)
print("Depart time", depart_time)
#plt.hist(depart_time, normed=True)
#plt.plot(arrival_time)
#plt.plot(depart_time)
#plt.show()

initial_state=np.random.uniform(0.8, 4.0, size=(total_vehicles,))
required_energy=np.random.uniform(2.0, 6.0, size=(total_vehicles,))
final_energy = np.min((initial_state+required_energy, np.ones_like(initial_state)*battery_capacity), axis=0)
required_energy = np.round(final_energy-initial_state, 2)

print("initial State", initial_state)
print("Required energy", required_energy)
print("Final state", final_energy)

initial_state_EDF=np.copy(initial_state)
u_mat=np.zeros((total_vehicles, num_steps), dtype=float)


#-5 to avoid computation infeasibility at this time
for t in range(np.int(arrival_time[0])+1, num_steps-5):
    power_budget=power_capacity #Change this for variable case

    print("Current time", t)

    #Firstly get the states
    print("current number of arrived cars", (arrival_time < t).sum())
    vehicle_ending_index = (arrival_time < t).sum()
    step_initial_SOC = np.copy(initial_state_EDF[:vehicle_ending_index])
    depart_schedule=np.copy(depart_time[:vehicle_ending_index])
    u_val=np.zeros_like(step_initial_SOC)
    index=np.argsort(depart_schedule) #sort the departure time
    charging_sessions=0

    while power_budget>=0:
        if depart_schedule[index[charging_sessions]] >= t:
            available_charging=np.minimum(max_power, power_budget)
            u_val[index[charging_sessions]] = np.maximum(np.minimum(available_charging, final_energy[index[charging_sessions]]-step_initial_SOC[index[charging_sessions]]),0)

        power_budget-=u_val[index[charging_sessions]]
        charging_sessions+=1

        if charging_sessions>=vehicle_ending_index:
            break



    #print("U SOC", np.round(u_val[:,0],2))
    print("SUM EDF", np.sum(u_val))
    updated_val=np.minimum(u_val, np.ones_like(u_val)*max_power - decay_rate * step_initial_SOC)
    #print("U after MPC cut", updated_val)
    print("SUM EDF Cut", np.sum(updated_val))
    initial_state_EDF[:vehicle_ending_index] += updated_val
    #print("SOC_states", np.round(initial_state_SOC[:vehicle_ending_index],2))
    u_mat[:vehicle_ending_index, t]=updated_val





df = pd.DataFrame(np.round(initial_state_EDF,2))
df.to_csv('Final_EDF_%s.csv'%decay_rate, index=False)
df = pd.DataFrame(np.round(u_mat,2))
df.to_csv('control_EDF_%s.csv'%decay_rate, index=False)
