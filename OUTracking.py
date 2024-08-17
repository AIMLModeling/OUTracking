import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import minimize
np.random.seed(seed=42)

# Parameters and Time Vector Initialization
N = 20000  # time steps
paths = 5000  # number of paths
T = 5
T_vec, dt = np.linspace(0, T, N, retstep=True)

# Vasicek Model Parameters
kappa = 3                # mean reversion coefficient
theta = 0.5              # long term mean
sigma = 0.5              # volatility coefficient
std_asy = np.sqrt(sigma**2 / (2 * kappa))  # asymptotic standard deviation

# Initial Simulation
X0 = 2
X = np.zeros((N, paths))
X[0, :] = X0
W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
for t in range(0, N - 1):
    X[t + 1, :] = theta + np.exp(-kappa * dt) * (X[t, :] - theta) + std_dt * W[t, :]
# Performs an Ordinary Least Squares (OLS) regression to estimate the parameters
# kappa, theta, and sigma from the simulated data.
X_T = X[-1, :]  # values of X at time T
X_1 = X[:, 1]  # a single path
XX = X_1[:-1]
YY = X_1[1:]
beta, alpha, _, _, _ = ss.linregress(XX, YY)  # OLS
kappa_ols = -np.log(beta) / dt
theta_ols = alpha / (1 - beta)
res = YY - beta * XX - alpha  # residuals
std_resid = np.std(res, ddof=2)
sig_ols = std_resid * np.sqrt(2 * kappa_ols / (1 - beta**2))

print("OLS theta = ", theta_ols)
print("OLS kappa = ", kappa_ols)
print("OLS sigma = ", sig_ols)

# Simulation of Ornstein-Uhlenbeck (OU) Paths
X0 = 2
X = np.zeros((N, paths))
X[0, :] = X0
W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
for t in range(0, N - 1):
    X[t + 1, :] = theta + np.exp(-kappa * dt) * (X[t, :] - theta) + std_dt * W[t, :]
    
sig_eta = std_resid
var_eta = sig_eta**2  # Process Noise (error of the true state process)
sig_eps = 0.1
var_eps = sig_eps**2  # Measurement Noise (error of the measurement)
np.random.seed(seed=42)
eps = ss.norm.rvs(loc=0, scale=sig_eps, size=N)  # additional gaussian noise
eps[0] = 0
Y_1 = X_1 + eps  # process + noise = measurement process
fig = plt.figure(figsize=(13, 4))
plt.plot(T_vec, X_1, linewidth=0.5, alpha=1, label="True process", color="#1f77b4")
plt.plot(T_vec, theta * np.ones_like(T_vec), label="Long term mean", color="#d62728")
plt.plot(T_vec, Y_1, linewidth=0.3, alpha=0.5, label="Measurement process = true + noise", color="#1f77b4")
plt.legend()
plt.title("OU process plus some noise")
plt.xlabel("time")
plt.show()
# Implements the Kalman filter algorithm to estimate the state of the OU process
# from noisy observations.
def Kalman(Y, x0, P0, alpha, beta, var_eta, var_eps):
    """Kalman filter algorithm for the OU process."""

    N = len(Y)  # length of measurements
    xs = np.zeros_like(Y)  # Initialization
    Ps = np.zeros_like(Y)

    x = x0
    P = P0  # initial values of h and P
    log_2pi = np.log(2 * np.pi)
    loglikelihood = 0  # initialize log-likelihood
    for k in range(N):
        # Prediction step
        x_p = alpha + beta * x  # predicted h
        P_p = beta**2 * P + var_eta  # predicted P
        # auxiliary variables
        r = Y[k] - x_p  # residual
        S = P_p + var_eps
        KG = P_p / S  # Kalman Gain
        # Update step
        x = x_p + KG * r
        P = P_p * (1 - KG)
        loglikelihood += -0.5 * (log_2pi + np.log(S) + (r**2 / S))
        xs[k] = x
        Ps[k] = P
    return xs, Ps, loglikelihood
# Prepares training and testing datasets from the noisy observations.
skip_data = 1000
training_size = 5000
train = Y_1[skip_data : skip_data + training_size]
test = Y_1[skip_data + training_size :]

# Estimates the parameters alpha, beta, var_eta(Process Noise), and var_eps(Measurement Noise)
# by maximizing the likelihood using the Kalman filter.
guess_beta, guess_alpha, _, _, _ = ss.linregress(train[1:], train[:-1])
guess_var_eps = np.var(train[:-1] - guess_beta * train[1:] - guess_alpha, ddof=2)
def minus_likelihood(c):
    """Returns the negative log-likelihood"""
    _, _, loglik = Kalman(train, X0, 10, c[0], c[1], c[2], c[3])
    return -loglik


result = minimize(
    minus_likelihood,
    x0=[guess_alpha, guess_beta, 0.01, guess_var_eps],
    method="L-BFGS-B",
    bounds=[[-1, 1], [1e-15, 1], [1e-15, 1], [1e-15, 1]],
    tol=1e-12,
)
kalman_params = result.x
alpha_KF = kalman_params[0]
beta_KF = kalman_params[1]
var_eta_KF = kalman_params[2]
var_eps_KF = kalman_params[3]
print(f"Error in estimation of alpha = {(np.abs(alpha-alpha_KF)/alpha *100).round(2)}%")
print(f"Error in estimation of beta = {(np.abs(beta-beta_KF)/beta *100).round(4)}%")
print(f"Error in estimation of var eta = {(np.abs(var_eta-var_eta_KF)/var_eta *100).round(2)}%")
print(f"Error in estimation of var eps = {(np.abs(var_eps-var_eps_KF)/var_eps *100).round(2)}%")
# Implements a Kalman smoother for better state estimation by considering 
# both past and future observations.
def Smoother(Y, x0, P0, alpha, beta, var_eta, var_eps):
    """Kalman smoother"""
    xs, Ps, _ = Kalman(Y, x0, P0, alpha, beta, var_eta, var_eps)
    xs_smooth = np.zeros_like(xs)
    Ps_smooth = np.zeros_like(Ps)
    Cs_smooth = np.zeros_like(Ps)
    C = np.zeros_like(Ps)
    xs_smooth[-1] = xs[-1]
    Ps_smooth[-1] = Ps[-1]
    K = (beta**2 * Ps[-2] + var_eta) / (beta**2 * Ps[-2] + var_eta + var_eps)
    Cs_smooth[-1] = Ps[-1]
    Cs_smooth[-2] = beta * (1 - K) * Ps[-2]

    for k in range(len(xs) - 2, -1, -1):
        C[k] = beta * Ps[k] / (beta**2 * Ps[k] + var_eta)
        xs_smooth[k] = xs[k] + C[k] * (xs_smooth[k + 1] - (alpha + xs[k] * beta))
        Ps_smooth[k] = Ps[k] + C[k] ** 2 * (Ps_smooth[k + 1] - (beta**2 * Ps[k] + var_eta))
        if k == (len(xs) - 2):
            continue
        Cs_smooth[k] = C[k] * Ps[k + 1] + C[k] * C[k + 1] * (
            Cs_smooth[k + 1] - beta * Ps[k + 1]
        )  # covariance x(k) and x(k+1)
    return xs_smooth, Ps_smooth, Cs_smooth
# Visualizes the true process, noisy measurements, Kalman filter estimates, and smoother estimates.
x_tmp, P_tmp, _ = Kalman(train, 1, 10, alpha_KF, beta_KF, var_eta_KF, var_eps_KF)  # to get initial values for KF
xs, Ps, _ = Kalman(test, x_tmp[-1], P_tmp[-1], alpha_KF, beta_KF, var_eta_KF, var_eps_KF)
x_smooth, P_smooth, _ = Smoother(test, x_tmp[-1], P_tmp[-1], alpha_KF, beta_KF, var_eta_KF, var_eps_KF)
V_up = xs + np.sqrt(Ps)  # error up bound
V_down = xs - np.sqrt(Ps)  # error down bound
fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(
    T_vec[skip_data + training_size :], xs, linewidth=0.5, alpha=1, label="Optimal state estimate", color="#1f77b4"
)
ax1.plot(T_vec[skip_data + training_size :], theta * np.ones_like(xs), label="Long term mean", color="#d62728")
ax1.plot(
    T_vec[skip_data + training_size :],
    Y_1[skip_data + training_size :],
    linewidth=0.6,
    alpha=0.5,
    label="Process + noise",
    color="#1f77b4",
)
ax1.plot(
    T_vec[skip_data + training_size :],
    x_smooth,
    linewidth=0.5,
    alpha=1,
    label="Smoothed state estimate",
    color="purple",
)
ax1.fill_between(
    x=T_vec[skip_data + training_size :],
    y1=V_up,
    y2=V_down,
    alpha=0.7,
    linewidth=2,
    color="seagreen",
    label="Kalman error: $\pm$ 1 std dev ",
)
ax1.legend()
ax1.set_title("State estimation of the OU process")
ax1.set_xlabel("time")
ax2.plot(T_vec[skip_data + training_size :], xs, linewidth=1, alpha=1, label="Optimal state estimator", color="blue")
ax2.plot(T_vec[skip_data + training_size :], theta * np.ones_like(xs), label="Long term mean", color="#d62728")
ax2.plot(
    T_vec[skip_data + training_size :],
    Y_1[skip_data + training_size :],
    linewidth=0.5,
    alpha=0.8,
    label="Process + noise",
    color="#1f77b4",
)
ax2.plot(
    T_vec[skip_data + training_size :], x_smooth, linewidth=1, alpha=1, label="Smoothed state estimate", color="purple"
)
ax2.fill_between(
    x=T_vec[skip_data + training_size :],
    y1=V_up,
    y2=V_down,
    alpha=0.7,
    linewidth=2,
    color="seagreen",
    label="Kalman error: $\pm$ 1 std dev ",
)
ax2.set_xlim(1.99, 2.15)
ax2.set_ylim(0.2, 0.65)
ax2.legend()
ax2.set_title("Zoom")
ax2.set_xlabel("time")
plt.show()
print("Test set, mean linear error Kalman: ", np.linalg.norm(xs - X_1[skip_data + training_size :], 1) / len(xs))
print("Average standard deviation of the estimate: ", np.mean(np.sqrt(Ps)))
print("Test set, MSE Kalman: ", np.linalg.norm(xs - X_1[skip_data + training_size :], 2) ** 2 / len(xs))
print("Test set, MSE Smoother: ", np.linalg.norm(x_smooth - X_1[skip_data + training_size :], 2) ** 2 / len(x_smooth))

# Refines parameter estimates iteratively using the Kalman smoother until convergence.
N_max = 1000  # number of iterations
err = 0.001  # error in the parameters
NN = len(train)
alpha_SS = guess_alpha  # initial guess
beta_SS = guess_beta  # initial guess
var_eps_SS = guess_var_eps  # initial guess
var_eta_SS = 0.1  # initial guess
x_start = 1  # initial guess
P_start = 10  # initial guess
for i in range(N_max):
    a_old = alpha_SS
    b_old = beta_SS
    eta_old = var_eta_SS
    eps_old = var_eps_SS

    x_sm, P_sm, C_sm = Smoother(train, x_start, P_start, alpha_SS, beta_SS, var_eta_SS, var_eps_SS)

    AA = np.sum(P_sm[:-1] + x_sm[:-1] ** 2)  # A
    BB = np.sum(C_sm[:-1] + x_sm[:-1] * x_sm[1:])  # B
    CC = np.sum(x_sm[1:])  # C
    DD = np.sum(x_sm[:-1])  # D

    alpha_SS = (AA * CC - BB * DD) / (NN * AA - DD**2)
    beta_SS = (NN * BB - CC * DD) / (NN * AA - DD**2)
    var_eta_SS = (
        np.sum(
            P_sm[1:]
            + x_sm[1:] ** 2
            + alpha_SS**2
            + P_sm[:-1] * beta_SS**2
            + (x_sm[:-1] * beta_SS) ** 2
            - 2 * alpha_SS * x_sm[1:]
            + 2 * alpha_SS * beta_SS * x_sm[:-1]
            - 2 * beta_SS * C_sm[:-1]
            - 2 * beta_SS * x_sm[1:] * x_sm[:-1]
        )
        / NN
    )
    var_eps_SS = np.sum(train**2 - 2 * train * x_sm + P_sm + x_sm**2) / (NN + 1)

    if (
        (np.abs(a_old - alpha_SS) / a_old < err)
        and (np.abs(b_old - beta_SS) / b_old < err)  # iterate until there is no improvement
        and (np.abs(eta_old - var_eta_SS) / eta_old < err)
        and (np.abs(eps_old - var_eps_SS) / eps_old < err)
    ):
        print("Number of iteration: ", i)
        break
if i == N_max - 1:
    print("Maximum number of iterations reached ", i + 1)
# Compares the estimated parameters from the Kalman filter and smoother with the true parameters.
print(f"Value of estimated alpha = {alpha_SS} vs real alpha = {alpha}")
print(f"Value of estimated beta = {beta_SS} vs real beta = {beta}")
print(f"Value of estimated var_eta = {var_eta_SS} vs real var_eta = {var_eta}")
print(f"Value of estimated var_eps = {var_eps_SS} vs real var_eps = {var_eps}")
