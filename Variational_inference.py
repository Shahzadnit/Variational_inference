import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os
import shutil
from collections import deque


from sklearn.datasets import make_spd_matrix
from random import  uniform
from scipy.linalg import cholesky, det, solve_triangular
################################################################################################################
def data_generation_2d(num_of_sample = 100, mean = 5, num_of_component = 3):
    sap = 3
    t_means = []
    for i in range(num_of_component):
        t_means.append([uniform(mean-sap, mean+sap), uniform(mean-sap, mean+sap)])
    # for each cluster center, create a Positive semidefinite convariance matrix
    t_covs = []
    for s in range(len(t_means)):
        t_covs.append(make_spd_matrix(2))
    X = []
    for mean, cov in zip(t_means,t_covs):
        x = np.random.multivariate_normal(mean, cov, num_of_sample)
        X += list(x)
    
    X = np.array(X)
    np.random.shuffle(X)
    return X

def estimate_prior_parameters(data):
    # Compute the mean
    mu_prior = np.mean(data, axis=0)

    # Compute the covariance matrix
    covariance_matrix = np.cov(data, rowvar=False)

    return mu_prior, covariance_matrix


# Define the ELBO as a function of the variational parameters
def elbo(X, mu, sigma, mu_p, sigma_p):
    # Compute log-likelihood term    
    N, D = X.shape
    log_likelihood = np.sum(-0.5 * np.log((2 * np.pi)**D * np.linalg.det(sigma)) - 0.5 * np.sum((X - mu) @ np.linalg.inv(sigma) * (X - mu), axis=1))

    
    # Compute KL divergence term
    kl_divergence = 0.5 * (np.trace(np.linalg.inv(sigma_p) @ sigma) + np.dot((mu_p - mu).T, np.linalg.inv(sigma_p) @ (mu_p - mu)) - num_latent_dims + np.log(np.linalg.det(sigma_p) / np.linalg.det(sigma)))
    
    # Compute ELBO
    elbo = log_likelihood - kl_divergence
    
    return elbo
def compute_gradient_ELBO_sigma(X, mu, sigma, mu_prior, sigma_prior):
    N, D = X.shape
    sigma_inv = np.linalg.inv(sigma)
    
    # Gradient of log-likelihood term
    gradient_LL = np.zeros_like(sigma)
    for i in range(N):
        x_i = X[i]
        gradient_LL += -0.5 * ((sigma_inv).T - np.dot(sigma_inv,np.dot(np.dot((x_i - mu),(x_i - mu).T),sigma_inv)))
    # Gradient of KL divergence term
    sigma_prior_inv = np.linalg.inv(sigma_prior)
    gradient_KL = 0.5 * (sigma_prior_inv.T - sigma_inv)
    # Gradient of ELBO
    gradient_ELBO = gradient_LL - gradient_KL
    return gradient_ELBO

def compute_gradient_ELBO_mu(X, mu_q, log_sigma_q,mu_p,sigma_p):
    sigma_p_inv = np.linalg.inv(sigma_p)
    grad_mu = np.sum(np.dot((X - mu_q), np.linalg.inv(log_sigma_q)), axis=0) + np.dot(sigma_p_inv ,(mu_p - mu_q))
    return grad_mu

def update(mu_q,log_sigma_q,i,elbo):
    # Compute the contour values for the current frame
    predicted_dist = np.zeros_like(X_grid)
    for j in range(n_hidden):
        predicted_dist += multivariate_normal.pdf(grid, mean=mu_q, cov=log_sigma_q)
        
    ax1.clear()
    # ax.figure(figsize=(16, 12))
    ax1.contourf(X_grid, Y_grid, predicted_dist, cmap='Blues')
    # ax1.contour(X_grid, Y_grid, predicted_dist, levels=25, colors='g', linewidths=1.5, alpha=0.5, label='Estimated Distribution')
    ax1.scatter(X[:, 0], X[:, 1], c='r', s=1 ,label='Data')
    ax1.scatter(mu_q[0], mu_q[1], c='b', marker='^', label='Approximate Posterior')
    
    ax1.set_title("Predicted Distribution at epoch %d"%i)
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    
    ax2.clear()
    que.append(elbo)
    ax2.plot(que)
    
    ax2.set_title("elbo at epoch %d"%i)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Elbo")
#############################################################################


# Generate a synthetic 2D dataset
# np.random.seed(42)
n_samples = 500
# X = np.random.normal(11, 0.9, (n_samples, 2))

# Variational Inference
n_iter = 3000
n_hidden = 3  # Number of hidden units

X = data_generation_2d(num_of_sample=(n_samples//n_hidden),mean= 15, num_of_component = n_hidden)

# prior distribution

mu_p, sigma_p = estimate_prior_parameters(X)



# Initialize variational parameters
num_latent_dims = 2
mu_q = np.zeros(num_latent_dims)
log_sigma_q = np.eye( num_latent_dims)

########################################
fig = plt.figure(figsize=(20, 20))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# fig, ax = plt.subplots()
x = np.linspace(-25, 25, 100)
y = np.linspace(-25, 25, 100)
X_grid, Y_grid = np.meshgrid(x, y)
grid = np.dstack((X_grid, Y_grid))
que = deque(maxlen = n_iter)



    
##########################################
learning_rate = 0.0001
elbo_history = []

# Optimization loop
for i in range(n_iter):
    # Sample from the variational posterior
    # Compute gradients of the ELBO
    ############# Del(Elbo)/Del(mu) #############
    grad_mu = compute_gradient_ELBO_mu(X, mu_q, log_sigma_q,mu_p,sigma_p)
    ############# Del(Elbo)/Del(sigma) #############
    grad_sigma = compute_gradient_ELBO_sigma(X, mu_q, log_sigma_q, mu_p, sigma_p)
    
    # Update variational parameters
    mu_q += learning_rate * grad_mu
    log_sigma_q += learning_rate * grad_sigma
    
    # Compute the ELBO
    current_elbo = elbo(X, mu_q, log_sigma_q,mu_p, sigma_p)
    elbo_history.append(current_elbo)
    
    if i % 10 == 0:
        print(f"Iteration {i}: ELBO = {current_elbo}")
        
    update(mu_q,log_sigma_q,i,current_elbo)
    plt.pause(0.001)
    
plt.show()

