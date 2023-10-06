import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def fobj(x0, A, b):
    N, d = A.shape
    fp = 0
    for i in range(N):
        fp += 0.5 * (A[i, :].dot(x0) - b[i]) ** 2
    return fp


def byzantine_filter(values, f, temp_xavg):
    n = values.shape[0]  # number of processors
    m = n - 2 * f - 1  # number of correct processors required to achieve consensus
    d = np.zeros(n)  # distance matrix

    for i in range(n):
        d[i] = np.linalg.norm(values[i, :] - temp_xavg)

    decision = np.argsort(d)[:m]
    return decision


np.random.seed(209920)  # Seed the random number generator

nSimulation = 1
sNetwork = [50]
nGRound = 50
nLRound = 2
sim = 0
N = sNetwork[sim]
d = 10

A = 0.8 * np.ones((N, d))
b = 0.5 * np.ones((N, 1))
maxa = np.max(A) / 2

x0 = np.zeros((d, 1))

options = {'ftol': 1e-6, 'maxfev': 1500 * len(x0)}
res = minimize(lambda x: fobj(x, A, b), x0, options=options)
x_opt = res.x
f_val = res.fun

x_cell = [np.zeros((d, 1)) for _ in range(N)]
y_cell = [np.zeros((d, 1)) for _ in range(N)]
err_x = [0]

for i in range(N):
    err_x[0] += np.linalg.norm(x_cell[i] - x_opt, 2)

rr = 1
nByzantine = [0, 2, 4, 6, 8]

for k in range(len(nByzantine)):
    for observations in range(50):
        nIters = 1
        GIters = 1
        LIters = 1

        f = [fobj(np.zeros((d, 1)), A, b)]
        f_err = [np.abs(f[0] - f_val)]
        x_cent = np.zeros((d, 1))
        temp_xavg = 0

        allagentindex = np.arange(1, N + 1)
        ByzantineIndex = np.random.permutation(N)[:nByzantine[k]]
        HonestIndex = np.setdiff1d(allagentindex, ByzantineIndex)
        p = 1

        while GIters < nGRound:
            GIters += 1
            err_x.append(0)
            f.append(0)
            alpha = 0.0055 / (GIters - 1)
            LIters = 1
            tempi = temp_xavg.copy()

            while LIters <= nLRound:
                nIters += 1
                LIters += 1

                for i in range(N):
                    gradi = (A[i].dot(x_cell[i]) - b[i]) * A[i].T
                    noise = np.random.normal(0, 5, gradi.shape)
                    tempi -= alpha * (gradi + noise)
                    x_cell[i] = tempi.copy()

                    if nByzantine[k] != 0 and i + 1 in ByzantineIndex:
                        bias = 1
                        tempi = tempi - alpha * (gradi + bias + noise)
                        x_cell[i] = tempi.copy()

                    if LIters == nLRound:
                        x_hist[i] = tempi.T
                    p += 1

            f2 = nByzantine[k]
            if f2 != 0:
                decision = byzantine_filter(x_hist, f2, temp_xavg)
            else:
                decision = np.arange(1, N + 1)
            temp_xavg = np.sum(x_hist[decision], axis=0) / len(decision)

            x_avg[GIters - 1, :] = temp_xavg
            f[GIters - 1] = max(f[GIters - 1], fobj(temp_xavg, A, b))

        for j in range(x_avg.shape[0]):
            err[j, observations] = np.linalg.norm(x_avg[j] - x_opt)
            f_err[j, observations] = np.abs(f[j] - f_val) / np.max(f)

    time = np.arange(1, GIters + 1)

    plt.figure(k)
    plt.subplot(2, 1, 1)
    if nByzantine[k] == 0:
        plt.plot(time, np.mean(f_err, axis=1), '-o', markerindices=np.arange(0, len(time), 2),
                 linewidth=2, markersize=3)
    else:
        plt.plot(time, np.mean(f_err, axis=1), '-o', markerindices=np.arange(0, len(time), 2),
                 linewidth=2, markersize=3)
        plt.grid(True)

    plt.ylim(0, 1)
    plt.xlabel('Iterations', interpreter='latex')
    plt.ylabel('Error', interpreter='latex')
    plt.xticks(fontweight='bold', fontsize=25)
    plt.yticks(fontweight='bold', fontsize=25)

legend_labels = ['No Faulty agents', 'Faulty agents = 2', 'Faulty agents = 4',
                 'Faulty agents = 8', 'Faulty agents = 10']
plt.legend(legend_labels, interpreter='latex', loc='northeast')
plt.tight_layout()
plt.show()
