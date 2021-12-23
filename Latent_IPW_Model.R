# Read in the IPW Matrix from the treatment model
ipw_matrix <- read_rds("data/ipw_matrix.rds")

# Load the Mosquito Nets Data
nets <- read_rds("data/mosquito_nets.rds")

"data {
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
  matrix[N, Kc] Xc;  // Centered Version of X without an Intercept
  vector[Kc] means_X;  // Column means of X before centering
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
}

parameters {
  real alpha; // Population-Level Intercept
  vector[Kc] beta; // Population-Level Effects
  real<lower=0> sigma;  // Dispersion Parameter
  vector<lower=0>[N] weights_z; // Standardized IPW Weights
}

transformed parameters {
  vector[N] weights_tilde; // Latent IPW Weights
  weights_tilde = gamma_w + delta_w .* weights_z;
}

model {
  // Likelihood
  vector[N] mu = alpha + Xc * beta;
  for (n in 1:N) {
    target += (normal_lpdf(Y[n] | mu[n], sigma)) * weights_tilde[n];
  }
  // Sampling the Weights
  target += exponential_lpdf(weights_z | 1);
  // Priors for the model parameters
  target += student_t_lpdf(alpha | 3, 0, 2.5);
  target += normal_lpdf(beta | 0, 3);
  target += student_t_lpdf(sigma | 3, 0, 10) - 1 * student_t_lccdf(0 | 3, 0, 10);
}

generated quantities {
  // Actual Population-Level Intercept
  real Intercept = alpha - dot_product(means_X, beta);
  // Average Treatment Effect
  real mu_treated = (Intercept + beta[1]) - Intercept;
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = (normal_lpdf(Y[n] | alpha + Xc[n] * beta, sigma)) * weights_tilde[n];
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
  dir = "C:/Users/ajn0093/Dropbox/Bayesian-IPW/analyses/models/latent-ipw",
  basename = "Test_Weights_Model"
)

# Generate a cmdstan model with opencl
cmdstan_test <- cmdstan_model(
  stan_file = stan_model_code,
  dir = "C:/Users/ajn0093/Dropbox/Bayesian-IPW/analyses/models/latent-ipw",
  force_recompile = T
)

# Fit the model with OpenCL
cmdstan_test_fit <- cmdstan_test$sample(
  data = stan_data_ls, # A list object to pass the data to Stan
  seed = 1234, # Random Number Seed for Reproducibility
  output_dir = "C:/Users/ajn0093/Dropbox/Bayesian-IPW/analyses/models/latent-ipw/",
  chains = 6, # Number of chains to run
  parallel_chains = 6,
  iter_warmup = 3000, # Warmup Iterations
  iter_sampling = 3000, # Sampling Iterations
  refresh = 50
)

# Calculate Treated and Untreated Units
treats <- cmdstan_test_fit$draws(variables = c("Intercept", "beta", "mu_treated", "sigma")) %>% 
  # Tidy data frame of draws
  tidy_draws()

ate_plot <- ggplot(treats, aes(x = mu_treated)) +
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
