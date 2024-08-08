import numpy as np
import matplotlib.pyplot as plt

# PhC parameters
d1 = 200e-9
d2 = 200e-9
d = d1 + d2
n1 = np.sqrt(13)
n2 = np.sqrt(12)

# Input light parameters
kz = np.linspace(0, 2 * np.pi / d, 100)
c = 3e8

# Decomposition constants
N = 1  # This chooses the number of bands

# Create matrix and solve eigenvalue equation
w2 = np.zeros((int(2*N + 1), len(kz)))  # Set up storage array

for kk in range(len(kz)):
    # Make matrix using for loop
    M = np.zeros((int(2*N + 1), int(2*N + 1)))  # Added semicolon to suppress output
    for nn in range(-int(N), int(N) + 1):
        for pp in range(-int(N), int(N) + 1):
            Fn = (d1 / d) * (1 / n1**2 - 1 / n2**2) * np.sinc((pp - nn) * d1 / d)
            if (pp - nn) == 0:
                Fn = Fn + 1 / n2**2
            M[nn + int(N), pp + int(N)] = (2 * np.pi / d * nn + kz[kk])**2 * Fn

    # Find and sort the eigenvalues
    eigenvalues = np.sort(np.linalg.eigvalsh(M))
    w2[:, kk] = eigenvalues

# Plot band diagram
plt.figure(figsize=(8, 6))
for i in range(w2.shape[0]):
    plt.plot(kz * d, np.sqrt(w2[i, :]) * d, label=f'Band {i+1}')

plt.plot(kz * d, np.abs(kz) * d, 'k-', linewidth=2, label='Light Line')
plt.xlabel('$k_z d$', fontsize=14)
plt.ylabel('$\omega d / c$', fontsize=14)
plt.ylim([0, 2 * np.pi])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
