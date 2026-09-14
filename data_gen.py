

import cr3bp
import scipy as sp
import numpy as np




# 35 mu values from e-5 to 0.02 sampled logarithmically, 15 values from 0.02 to 0.5 sampled linearly
mu_log = np.logspace(np.log10(1e-5), np.log10(0.02), 35)
mu_lin = np.linspace(0.02, 0.5, 15)
mu_values = np.unique(np.concatenate((mu_log, mu_lin)))


def gen_ic(mu,n):
    ic_array = []
    x1,y1 = -mu,0
    x2,y2 = 1-mu,0  # positions of objects with mass
    for _ in range(n):
        x = np.random.uniform(-1.2,1.2)
        y = np.random.uniform(-1.2,1.2) # range ensures lagrange points are in

        vx = np.random.uniform(-1,1)
        vy = np.random.uniform(-1,1) # range makes sure massles object doesnt escape

        r1 = np.sqrt((x - x1)**2 + (y - y1)**2)
        r2 = np.sqrt((x - x2)**2 + (y - y2)**2)
        if r1 < 0.05 or r2 < 0.05: # check if i.c is close to either object

            while r1 < 0.05 or r2 < 0.05:
                x = np.random.uniform(-1.2,1.2)
                y = np.random.uniform(-1.2,1.2)
                r1 = np.sqrt((x - x1)**2 + (y - y1)**2)
                r2 = np.sqrt((x - x2)**2 + (y - y2)**2)
        ic_array.append([x,y,0,vx,vy,0])
    return np.array(ic_array)


alldata = []
for mu in mu_values:
    ic_array = gen_ic(mu,20)
    eoms = cr3bp.EOMConstructor(mu)
    t_start = 0
    t_end = 2 * np.pi # one full rotation
    t_points = np.linspace(t_start, t_end, 1000) # dt = 0.006
    for ic in ic_array:
        solution = sp.integrate.solve_ivp(eoms, [t_start, t_end], ic, t_eval=t_points, atol=1e-12, rtol=1e-12)
        data = solution.y.T # transpose the horrible (6,1000) shape to (1000,6)
        mu_column = np.full((len(data), 1), mu) # add a mu column
        t_column = solution.t.reshape(-1, 1) # reshape t column aswell
        combined = np.hstack((data, t_column,mu_column)) # [x,y,z,vx,vy,vz,t,mu] , (1000,8)
        alldata.append(combined)

final_dataset = np.vstack(alldata)
np.save("cr3bp_dataset_final.npy", final_dataset)




