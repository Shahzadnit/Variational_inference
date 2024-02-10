# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import numpy as np

# # Generate random 2-dimensional data points
# np.random.seed(0)
# true_mean = np.array([3.0, 4.0])
# true_covariance = np.array([[1.0, 0.8], [0.8, 1.0]])
# num_samples = 100
# true_samples = np.random.multivariate_normal(true_mean, true_covariance, num_samples)

# # Convert data to PyTorch tensor
# true_samples_tensor = torch.tensor(true_samples, dtype=torch.float32)

# # Variational Distribution: 2-dimensional Gaussian
# class VariationalDistribution(nn.Module):
#     def __init__(self):
#         super(VariationalDistribution, self).__init__()
#         self.mu = nn.Parameter(torch.randn(2))  # Mean parameter
#         self.log_var = nn.Parameter(torch.randn(2))  # Log variance parameter

#     def forward(self):
#         return self.mu, torch.exp(self.log_var)

# # Geodesic Distance Calculation
# def geodesic_distance(mu1, log_var1, mu2, log_var2):
#     g11 = torch.exp(-log_var1)
#     g22 = torch.exp(-log_var2)
#     delta_mu = mu1 - mu2
#     delta_log_var = log_var1 - log_var2
#     distance = torch.sqrt(g11 * delta_mu[0]**2 + g11 * delta_mu[1]**2 + g22 * delta_log_var[0]**2 + g22 * delta_log_var[1]**2)
#     return distance

# # ELBO Calculation with Geodesic Distance Penalty
# def elbo_loss(true_samples_tensor, mu, log_var, lambda_param=0.1):
#     variational_distribution = torch.distributions.MultivariateNormal(mu, torch.diag(torch.exp(log_var)))
#     log_likelihood = variational_distribution.log_prob(true_samples_tensor).sum(dim=0)  # Sum across batch elements (dim=0)
#     geodesic_dist = geodesic_distance(mu0, log_var0, mu, log_var)
#     elbo = log_likelihood - lambda_param * geodesic_dist
#     return -elbo.mean()  # Compute mean loss for the batch




# # Initialize variational parameters
# mu0 = torch.tensor([0.0, 0.0], requires_grad=False)
# log_var0 = torch.tensor([0.0, 0.0], requires_grad=False)

# # Initialize variational distribution
# var_dist = VariationalDistribution()
# optimizer = optim.Adam(var_dist.parameters(), lr=0.01)

# # Training loop
# num_epochs = 1000
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     mu, log_var = var_dist()
#     loss = elbo_loss(true_samples_tensor, mu, log_var)
#     loss.backward()
#     optimizer.step()

#     # Plotting the approximate distribution dynamically
#     if (epoch + 1) % 50 == 0:
#         plt.figure(figsize=(8, 6))
#         plt.scatter(true_samples[:, 0], true_samples[:, 1], color='b', label='True Distribution')
#         plt.scatter(mu.detach().numpy()[0], mu.detach().numpy()[1], color='r', label='Approximate Distribution')
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         plt.title(f'Epoch {epoch + 1}')
#         plt.legend()
#         plt.show()

# # Final approximate distribution
# final_mu, final_log_var = var_dist()
# print(f'Final Approximate Mean: {final_mu.detach().numpy()}')
# print(f'Final Approximate Log Variance: {final_log_var.exp().detach().numpy()}')




import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Generate random 2-dimensional data points
np.random.seed(0)
true_mean = np.array([25.0, 30.0])
true_covariance = np.array([[5.0, 0.8], [0.8, 40.0]])
num_samples = 200
true_samples = np.random.multivariate_normal(true_mean, true_covariance, num_samples)

# Convert data to PyTorch tensor
true_samples_tensor = torch.tensor(true_samples, dtype=torch.float32)

# Variational Distribution: 2-dimensional Gaussian
class VariationalDistribution(nn.Module):
    def __init__(self):
        super(VariationalDistribution, self).__init__()
        self.mu = nn.Parameter(torch.randn(2))  # Mean parameter
        self.log_var = nn.Parameter(torch.randn(2))  # Log variance parameter

    def forward(self):
        return self.mu, torch.exp(self.log_var)

# Geodesic Distance Calculation
def geodesic_distance(mu1, log_var1, mu2, log_var2):
    g11 = torch.exp(-log_var1)
    g22 = torch.exp(-log_var2)
    delta_mu = mu1 - mu2
    delta_log_var = log_var1 - log_var2
    distance = torch.sqrt(g11 * delta_mu[0]**2 + g11 * delta_mu[1]**2 + g22 * delta_log_var[0]**2 + g22 * delta_log_var[1]**2)
    return distance

# ELBO Calculation with Geodesic Distance Penalty
def elbo_loss(true_samples_tensor, mu, log_var, lambda_param=0.1):
    variational_distribution = torch.distributions.MultivariateNormal(mu, torch.diag(torch.exp(log_var)))
    log_likelihood = variational_distribution.log_prob(true_samples_tensor).sum(dim=0)  # Sum across batch elements (dim=0)
    geodesic_dist = geodesic_distance(mu0, log_var0, mu, log_var)
    elbo = log_likelihood - lambda_param * geodesic_dist
    return -elbo.mean()  # Compute mean loss for the batch

# Initialize variational parameters
mu0 = torch.tensor([0.0, 0.0], requires_grad=False)
log_var0 = torch.tensor([0.0, 0.0], requires_grad=False)

# Initialize variational distribution
var_dist = VariationalDistribution()
optimizer = optim.Adam(var_dist.parameters(), lr=0.013)

# Training loop with continuous graph updates
num_epochs = 10000

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 6))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    mu, log_var = var_dist()
    loss = elbo_loss(true_samples_tensor, mu, log_var)
    loss.backward()
    optimizer.step()

    # Plotting the approximate distribution as a density heatmap
    if (epoch + 1) % 20 == 0:  # Update the plot every 10 epochs
        ax.clear()

        # Generate grid points for density estimation
        x = np.linspace(-50, 50, 100)
        y = np.linspace(-50, 50, 100)
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([X.ravel(), Y.ravel()]).T
        var_dist_distribution = torch.distributions.MultivariateNormal(mu, torch.diag(torch.exp(log_var)))
        density_values = var_dist_distribution.log_prob(torch.tensor(positions, dtype=torch.float32)).exp().detach().numpy()
        density_values = density_values.reshape(100, 100)

        # # Plot true samples
        # ax.scatter(true_samples[:, 0], true_samples[:, 1], color='r', label='True Distribution', alpha=0.5)

        # Plot approximate distribution as density heatmap
        ax.contourf(X, Y, density_values, cmap='jet', levels=50, alpha=0.8)
        # ax.contour(X, Y, density_values, levels=10, colors='g', linewidths=1.5, alpha=0.5)
        
         # Plot true samples
        ax.scatter(true_samples[:, 0], true_samples[:, 1], color='r', label='True Distribution', alpha=0.4)

        # Plot approximate mean
        ax.scatter(mu.detach().numpy()[0], mu.detach().numpy()[1], color='black', label='Approximate Mean', marker='x')

        # Set plot labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Epoch {epoch + 1}')
        ax.legend()

        # Pause for a short time to allow for the animation effect
        plt.pause(0.001)

# Final approximate distribution
final_mu, final_log_var = var_dist()
print(f'Final Approximate Mean: {final_mu.detach().numpy()}')
print(f'Final Approximate Log Variance: {final_log_var.exp().detach().numpy()}')

# Show the final plot
plt.show()
