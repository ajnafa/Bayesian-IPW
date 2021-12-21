#------Bayesian IPW Modeling for Causal Inference in a Bayesian Framework-------
#-Author: A. Jordan Nafa----------------------------Created: November 18, 2021-#
#-R Version: 4.1.0----------------------------------Revised: December 22, 2021-#

# Set Project Options----
options(
  digits = 4, # Significant figures output
  scipen = 999, # Disable scientific notation
  repos = getOption("repos")["CRAN"],
  brms.backend = "cmdstanr"
)

# Load the Required libraries----
pacman::p_load(
  "tidyverse",
  "data.table",
  "dtplyr",
  "datawizard",
  "brms",
  "cmdstanr",
  "tidybayes",
  "ggdag"
)

# Load the helper functions----
source("scripts/00_Helper_Functions.R")

# Base Directory for the model objects
models_dir <- "analyses/models/"

# Base Directory for the stan files
stan_dir <- "analyses/stan/"

# Read in the IPw Matrix from the treatment model
ipw_matrix <- read_rds("data/ipw_matrix.rds")

# Load the Mosquito Nets Data
nets <- read_rds("data/mosquito_nets.rds")

#------------------------------------------------------------------------------#
#-----------------------Specifying the Outcome Model----------------------------
#------------------------------------------------------------------------------#

# We can specify the outcome model almost entirely in brms as follows. First
# We define the formula and specify priors

# Specify the formula for the outcome model----
outcome_bf <- bf(
   malaria_risk | weights(weights) ~ net_num,
   family = brmsfamily("gaussian", "identity") # Normal Regression Model
)

# Specify some weakly informative priors for the model parameters----
outcome_priors <- 
   prior("student_t(3, 0, 2.5)", class = "Intercept") +
   prior("normal(0, 2.5)", class = "b")

# Next, we'll define the stanvars objects for the custom parameters
# and data needed for the IPW weights

# Defining the IPw Matrix in the data block
data_vars <- stanvar(
  x = as.matrix(ipw_matrix),
  name = "weights",
  scode = "int<lower = 1> L;
  matrix[N, L] IPW;",
  block = "data"
)

# Defining the weights matrix in the model block
model_vars <- stanvar(
  scode = "// Weights for the Current Iteration
  int M = get_iter();
  vector[N] weights = IPW[, M];",
  block = "model",
  position = "start"
)

# Defining the iteration update function in generated quantities
gq_vars <- stanvar(
  scode = "add_iter();",
  block = "genquant"
)

# Adding the iteration counter functions
fun_vars <- stanvar(
  scode = "void add_iter();                                                                 
  int get_iter();",
  block = "functions"
)

# Generate Stan Code for the Model
make_stancode(
  formula = outcome_bf,
  data = nets,
  prior = outcome_priors,
  sample_prior = "no",
  data2 = list(L = 8000, weights = as.matrix(ipw_matrix)),
  stanvars = c(data_vars, model_vars, gq_vars, fun_vars),
)

model_code <- "// generated with brms 2.16.2
functions {
  void add_iter();                                                                 
  int get_iter();
}
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  // vector<lower=0>[N] weights;  // model weights
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  int prior_only;  // should the likelihood be ignored?
  int<lower = 1> L;
  matrix[N, L] IPW;
}
transformed data {
  int Kc = K - 1;
  matrix[N, Kc] Xc;  // centered version of X without an intercept
  vector[Kc] means_X;  // column means of X before centering
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
}
parameters {
  vector[Kc] b;  // population-level effects
  real Intercept;  // temporary intercept for centered predictors
  real<lower=0> sigma;  // dispersion parameter
}
transformed parameters {
}
model {
  // Weights for the Current Iteration
  int M = get_iter();
  vector[N] weights = IPW[, M];
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = Intercept + Xc * b;
    for (n in 1:N) {
      target += weights[n] * (normal_lpdf(Y[n] | mu[n], sigma));
    }
  }
  // priors including constants
  target += normal_lpdf(b | 0, 2.5);
  target += student_t_lpdf(Intercept | 3, 0, 2.5);
  target += student_t_lpdf(sigma | 3, 0, 14.8)
  - 1 * student_t_lccdf(0 | 3, 0, 14.8);
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = Intercept - dot_product(means_X, b);
  add_iter();
}"

# Compile the stan model
ipw_outcome_model <- rstan::stan_model(
  model_name = "Outcome_IPW_Model",
  model_code = model_code,
  allow_undefined = TRUE, 
  includes = paste0('\n#include "', file.path(getwd(), 'analyses/stan/iterfuns.hpp'), '"\n'),
  save_dso = TRUE
)

# Define the data for the stan model
stan_data_ls <- list(
  N = nrow(nets),
  Y = nets$malaria_risk,
  K = 2,
  X = model.matrix(~ net_num, data = nets),
  L = 8000,
  IPW = as.matrix(ipw_matrix),
  prior_only = 0
)

# fit the model using rstan
samp_outcome_model <- rstan::sampling(
  ipw_outcome_model, 
  data = stan_data_ls, 
  chains = 6, 
  iter = 8000,
  seed = 1234,
  cores = 6
)

# @ todo write a brms wrapper function






