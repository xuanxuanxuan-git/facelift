import numpy as np
import matplotlib.pyplot as plt

def surface_1(x, y):
    return np.abs(((1 / (1 + np.exp(1 - 12 * (x + 1))) * 1 / (1 + np.exp(1 - 3 * (y + 1))) * 1 / (1 + np.exp((x))) *
                    (1 / (1 + np.exp(1 - 12 * ((y - 3) + 1))) * 1 / (1 + np.exp(1 - 3 * (y + 1))) * 1 / (
                                1 + 2 * np.exp(((y - 3))))) * 2.5 +
                    (1 / (1 + np.exp(1 - 12 * ((x - 2) + 1))) * 1 / (1 + np.exp(1 - (4 * y + 1))) * 1 / (
                                1 + np.exp(((x - 2)))) / 2 +
                     1 / (1 + np.exp(12 + 5 * x + 2 * y)) / 1.5)) - 1.1154647694013708e-07
                   ) / (0.7427793855415158 - 1.1154647694013708e-07
                        ))


def gen_points(x1, x2, lpert1, upert1, lpert2, upert2, num):
    for i in range(0, num):
        pert1 = np.random.uniform(lpert1, upert1)
        pert2 = np.random.uniform(lpert2, upert2)
        p = surface_1(x2 + pert2, x1 + pert1)
        if p > 1:
            p = 1
        if p < 0:
            p = 0
        colour = np.random.choice(['red', 'blue'], p=[p, 1 - p])
        plt.plot([x1 + pert1], [x2 + pert2], marker="o", markersize=4, markeredgecolor='black', markerfacecolor=colour,
                 alpha=0.25)