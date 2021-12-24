# Read in the IPW Matrix from the treatment model
ipw_matrix <- read_rds("data/ipw_matrix.rds")

# Load the Mosquito Nets Data
nets <- read_rds("data/mosquito_nets.rds")

"/* Bayesian Latent Inverse Probability Estimator with a Gaussian Likelihood
* Author: A. Jordan Nafa; Stan Version 2.28.1; Last Revised 12-23-2021 */
functions {
  // Weighted Log PDF of the Gaussian Pseudo-Likelihood
  real normal_ipw_lpdf(vector y, vector mu, real sigma, vector w_tilde, int N) {
    real weighted_term;
    weighted_term = 0.00;
    for (n in 1:N) {
      weighted_term = weighted_term + w_tilde[n] * (normal_lpdf(y[n] | mu[n], sigma));
    }
    return weighted_term;
  }
}
data {
  // Data for the outcome model
  int<lower = 0> N; // Total Number of Observations
  vector[N] Y; // Response Vector
  int<lower = 1> K; // Number of Population-Level Effects
  matrix[N, K] X; // Design Matrix for the Population-Level Effects
  // Data from the Design Stage Model
  int<lower = 1> IPW_N; // Number of Rows in the Weights Matrix
  matrix[IPW_N, N] IPW; // Matrix of IP Weights from the Design Stage Model
}
transformed data {
  // Data for the latent weights
  vector[N] gamma_w; // Mean of the IP Weights
  vector[N] delta_w; // SD of the IP Weights
  // Calculate the location and scale for each observation weight
  for (i in 1:N) {
    gamma_w[i] = mean(IPW[, i]);
    delta_w[i] = sd(IPW[, i]);
  }
  // Centering the Predictor Matrix
  int Kc = K - 1;
  matrix[N, Kc] Xc; // Centered version of X without an Intercept
  vector[Kc] means_X; // Column means of X before centering
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
}
parameters {
  real alpha; // Temporary Population-Level Intercept
  vector[Kc] beta; // Population-Level Effects
  real<lower=0> sigma;  // Dispersion Parameter
  vector<lower=0>[N] weights_z; // Standardized IPW Weights
}
transformed parameters {
  vector[N] w_tilde; // Latent IPW Weights
  // Compute the Latent IPW Weights
  w_tilde = gamma_w + delta_w .* weights_z;
}
model {
  // Likelihood
  vector[N] mu = alpha + Xc * beta;
  target += normal_ipw_lpdf(Y | mu, sigma, w_tilde, N);
  // Sampling the Weights
  target += exponential_lpdf(weights_z | 1);
  // Priors for the model parameters
  target += student_t_lpdf(alpha | 3, 0, 2.5);
  target += normal_lpdf(beta | 0, 5);
  target += student_t_lpdf(sigma | 3, 0, 10) - 1 * student_t_lccdf(0 | 3, 0, 10);
}
generated quantities {
  // Actual Population-Level Intercept
  real Intercept = alpha - dot_product(means_X, beta);
  // Calculating Treatment Effects
  real yhat_treated = Intercept + beta[1]*1; // Predictions for the Treated Units
  real yhat_untreated = Intercept + beta[1]*0; // Predictions for the Untreated Units
  real mu_ate = yhat_treated - yhat_untreated; // Average difference between groups
  // Pseudo Log-Likelihood
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = w_tilde[n] * (normal_lpdf(Y[n] | alpha + Xc[n] * beta, sigma));
  }
}
" -> stan_code

# Define the data for the stan model
stan_data_ls <- list(
  N = nrow(nets),
  Y = nets$malaria_risk,
  K = 2,
  X = model.matrix(~ net_num, data = nets),
  IPW_N = 8000,
  IPW = t(ipw_matrix)
)

stan_model_code <- write_stan_file(
  code = stan_code,
  dir = "C:/Users/adamj/Dropbox/Bayesian-IPW/analyses/models/latent-ipw",
  basename = "Test_Weights_Model"
)

# Generate a cmdstan model with opencl
cmdstan_test <- cmdstan_model(
  stan_file = stan_model_code,
  dir = "C:/Users/adamj/Dropbox/Bayesian-IPW/analyses/models/latent-ipw",
  force_recompile = T
)

# Fit the model with OpenCL
cmdstan_test_fit <- cmdstan_test$sample(
  data = stan_data_ls, # A list object to pass the data to Stan
  seed = 1234, # Random Number Seed for Reproducibility
  output_dir = "C:/Users/adamj/Dropbox/Bayesian-IPW/analyses/models/latent-ipw/",
  chains = 6, # Number of chains to run
  parallel_chains = 6,
  iter_warmup = 3000, # Warmup Iterations
  iter_sampling = 3000, # Sampling Iterations
  refresh = 50
)

cmdstan_test_fit$draws(variables = c("yhat_untreated", "yhat_treated", "mu_ate")) %>% 
  summarise_draws()

# # A tibble: 3 x 10
#variable        mean median    sd   mad    q5   q95  rhat ess_bulk ess_tail
#<chr>          <dbl>  <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>    <dbl>    <dbl>
#1 yhat_untreated 39.6   39.6  0.320 0.321  39.0 40.1   1.00   31796.   13017.
#2 yhat_treated   29.6   29.6  0.322 0.321  29.1 30.1   1.00   45111.   12784.
#3 mu_ate         -9.98  -9.98 0.455 0.456 -10.7 -9.24  1.00   36136.   12997.

# Calculate Treated and Untreated Units
treats <- cmdstan_test_fit$draws(
  variables = c("yhat_untreated", "yhat_treated", "mu_ate", "w_tilde")
  ) %>% 
  # Tidy data frame of draws
  tidy_draws()

ate_plot <- ggplot(treats, aes(x = mu_ate)) +
  # Add the gradient slab
  stat_halfeye(
    aes(slab_alpha = stat(pdf)),
    fill = "blue",
    fill_type = "gradient",
    show.legend = FALSE,
    point_interval = mean_qi
  ) +
  # Apply custom theme settings
  plot_theme(plot.margin = margin(5,5,5,5, "mm")) +
  # Add labels
  labs(
    y = "Density", 
    x = expression(paste("ATE", Delta))
  )

# Render the gradient plot to a file
ggsave(
  filename = "ATE_Graph_Latent_IPW.jpeg",
  dpi = "retina",
  plot = ate_plot,
  device = "jpeg",
  height = 8,
  width = 12,
  type = "cairo"
)

weights_plot <- treats %>% select(matches("w_tilde")) %>% 
  pivot_longer(everything()) %>% 
  ggplot(., aes(x = value)) +
  stat_halfeye(
    aes(slab_alpha = stat(pdf)),
    fill = "purple",
    show.legend = FALSE,
    point_interval = mean_qi
  ) +
  # Apply custom theme settings
  plot_theme(plot.margin = margin(5,5,5,5, "mm")) +
  # Add labels
  labs(
    y = "Density", 
    x = "Latent Weights"
  )

# Render the gradient plot to a file
ggsave(
  filename = "Weights_Graph_Latent_IPW.jpeg",
  dpi = "retina",
  plot = weights_plot,
  device = "jpeg",
  height = 8,
  width = 12,
  type = "cairo"
)

