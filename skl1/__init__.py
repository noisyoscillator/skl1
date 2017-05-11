import numpy as np

from .euler import integrate

def euler1_xv(x0, v0, gamma, m, f, D, dt, nsteps, nsubsteps):
    x = np.array(x0, copy=True)
    x_data = [x.copy()]
    v = np.array(v0, copy=True)
    v_data = [v.copy()]
    v_factor = m*np.sqrt(2*gamma*dt)
    force = f(x)
    for i in range(nsteps):
        for j in range(nsubsteps):
            x = x + v*dt + force*dt**2/2
            v = v + force*dt/2
            force = f(x)
            v = v + (force/2-gamma*v)*dt/m + \
                np.random.normal(loc=0, scale=v_factor, size=v.shape)
        x_data.append(x.copy())
        v_data.append(v.copy())
    return np.array(x_data), np.array(v_data)
