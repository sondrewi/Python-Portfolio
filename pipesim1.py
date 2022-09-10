import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import math
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
plt.style.use('seaborn-pastel')

# Set pipe parameters
length = 3.7332
sound_speed = 345
air_density = 1.204

# define gamma parameter as per equation
gamma = sound_speed / length

# define samplerate and length in seconds of output
samplerate = 44100
seconds = 5
timesteps = int(samplerate * seconds)

# array of time points at which I would like to approximate solution
time_points = np.linspace(0, seconds, timesteps, endpoint=False)

# Time step k
k = (1 / samplerate)

# spatial step satisfying CFL-condition (must not be less than gamma*k)
# Raise exception if violated
h = gamma * k

if (h < gamma * k):
    raise Exception("CFL-condition violated")

# Make a lambda variable
lambda_ = gamma * k / h

# Create array of spatial points at which I would like to approximate solution.
# The number of points must be an integer
spatial_count = int((1 / h) + 1)
spatial_points = np.linspace(0, 1, spatial_count)

# Define index number for last spatial point
N = spatial_count - 1

# Define Cross-sectional area S(x) in terms of radius
left_rad = 0.0549

# find reference cross-section S_0unscaled
S_0unscaled = ((left_rad)**2) * np.pi

# find area of cross section at different spatial points
# and scale by area at closed end
S = ((left_rad + spatial_points * 0.05)**2) * np.pi / S_0unscaled

# set virtual boundary points
S_min1 = S[1] - (0.05 * 4 * h * left_rad * np.pi / S_0unscaled)
S_Nplus1 = (0.05 * 4 * h * (left_rad + 0.05) *
            np.pi / S_0unscaled) + spatial_points[N - 1]


# Calculate mu_(xx)S
mu_xxS = np.zeros(spatial_count)
mu_xxS[0] = (S[1] + 2 * S[0] + S_min1) / 4
mu_xxS[N] = (S_Nplus1 + 2 * S[N] + S[N - 1]) / 4
for i in range(1, N):
    mu_xxS[i] = (S[i + 1] + 2 * S[i] + S[i - 1]) / 4

# Define parameters alpha_1, alpha_2
alpha_1 = (((2 * 0.6133)**2) * gamma)**(-1)
alpha_2 = length * np.sqrt(np.pi) / (0.6133 * np.sqrt(S_0unscaled * S[1]))

# define an input frequency
input_freq = 523.25


def u_in(t):
    # Input function with instability resulting from beating of reed
    # or lip on flue pipe. Assume particle velocity at x=0 is initially 0
    # and increasing up until time point t=0.3s
    if t <= 0.3:
        return (1 / (0.09)) * (t**2) * (np.sin(2 * np.pi * (input_freq) * t))
    else:
        return (np.sin(2 * np.pi * (input_freq) * t))


# Define matrix according to (27), (30), (31)
matrix = 2 * (1 - (lambda_**2)) * np.identity(spatial_count)
matrix[0, 1] = 2 * (lambda_**2)
for i in range(1, N):
    matrix[i, i - 1] = (lambda_**2) * (S[i] + S[i - 1]) / (2 * mu_xxS[i])
    matrix[i, i + 1] = (lambda_**2) * (S[i + 1] + S[i]) / (2 * mu_xxS[i])

# Define last row in matrix
tau = (lambda_**2) * h * (S_Nplus1 + S[N]) * \
    (alpha_1 + (alpha_2 * k)) + 2 * k * mu_xxS[N]
matrix[-1, -1] = 4 * k * (1 - (lambda_**2)) * (mu_xxS[N]) * (1 / tau)
matrix[-1, -2] = 4 * k * (lambda_**2) * mu_xxS[N] * (1 / tau)

# Define factor on u_in
u_in_factor = (lambda_**2) * h * (S_0unscaled + S_min1) / (mu_xxS[0])

# Coefficient on extra term for Psi_N_nmin1
Psi_N_nmin1_coeff = (lambda_**2) * h * (S_Nplus1 + S[N]) * alpha_1 * (1 / tau)

# Set initial conditions for the parameter Psi (time points 0 and 1).
Psi_x_nmin1 = np.zeros(spatial_count)
Psi_x_n = np.copy(Psi_x_nmin1)

# Initiate output at x=1. Output is measured in pressure deviation
# from some reference, p = -air-density * delta_(t dot)Psi
output = np.zeros(timesteps)

# Function that iteratively fills output array


def update(i):
    global Psi_x_nmin1, Psi_x_n, output, counter

    # calculate next time step as given by formula
    Psi_x_next = matrix @ (Psi_x_n.T) - Psi_x_nmin1
    Psi_x_next[0] -= u_in_factor * u_in(time_points[i])
    Psi_x_next[-1] += Psi_N_nmin1_coeff * Psi_x_nmin1[N]

    # calculate the pressure deviation from atm
    pressure_deviation = -air_density * (Psi_x_next - Psi_x_nmin1) / (2 * k)

    # append to output
    output[i] = pressure_deviation[N]

    # move indeces
    Psi_x_nmin1 = np.copy(Psi_x_n)
    Psi_x_n = np.copy(Psi_x_next)

    return pressure_deviation


# Run update function from time-point 1 to seconds*samplerate
for i in range(1, timesteps):
    update(i)

# Plot pressure profile at x=1 against time
plt.plot(time_points, output)
plt.xlabel("Time")
plt.ylabel("Pressure Deviation from Atm at x=1")
plt.show()

# Audio output at x=1:
# The final amplitude should span the signed 16-bit integers range [-2**15,2**15)
# iinfo returns the range for the int16 type, then max resolves to -> 2**15
print(np.max(output))
signal1 = output
max_value = np.max(abs(signal1))
amplitude = np.iinfo(np.int16).max
data = amplitude * signal1 / max_value
wavwrite("pipesim.wav", samplerate, data.astype(np.int16))

# display sample rate and samples
rate, data = wavread("pipesim.wav")
print('spatial_count:', spatial_count)
print('rate:', rate, 'Hz')
print('data is a:', type(data))
print('points are a:', type(data[0]))
print('data shape is:', data.shape)

# Play sound from wav file using relevant library
from playsound import playsound
file = '/Users/sondrew/Documents/jobb/Internship 2022/Summer Project/CODE/pipesim.wav'
playsound(file, block=True)
