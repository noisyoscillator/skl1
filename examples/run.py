#!/usr/bin/env python3
"""Execute a Langevin simulation"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--steps', type=int, default=2**11)
parser.add_argument('--gamma', type=float, default=1)
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt
import skl1
import scipy.signal

if args.seed is not None:
    seed = args.seed
else:
    import time
    seed = int((time.time()*1000) % 2**32)


def f(x):
    return 0

def g(v):
    return -args.gamma*v

x, v = 0, 0

interval = 10
dt = 0.01

x, v = skl1.integrate(x, v, 1, dt, interval, args.steps, f, g, seed)

N = len(v)
t = dt*interval*np.arange(N)

ax1 = plt.subplot(311)
plt.plot(t, x)
ax2 = plt.subplot(312, sharex=ax1)
plt.plot(t, v)

plt.subplot(313)

fft_cor = scipy.signal.fftconvolve(v, v[::-1])[N-1:]
fft_cor /= (N - np.arange(N))

idx_max = int(10/(args.gamma*interval*dt))

plt.plot(t[:idx_max], fft_cor[:idx_max], 'k-', lw=2)
plt.plot(t[:idx_max], np.exp(-args.gamma*t[:idx_max])/args.gamma)

plt.show()
