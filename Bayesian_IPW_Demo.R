#------Bayesian IPW Modeling for Causal Inference in a Bayesian Framework-------
#-Author: A. Jordan Nafa----------------------------Created: November 18, 2021-#
#-R Version: 4.1.0----------------------------------Revised: December 18, 2021-#

# Note: All analysis for the models were performed on a Windows 10 desktop 
# computer with a Ryzen 9 5900X CPU, 128GB of DDR4 Memory, and an Nvidia RTX 
# 3080TI GPU using cmdstan 2.8.2 as a backend via cmdstanr and brms.

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

#------------------------------------------------------------------------------#
#-----------------------------Data Preparation----------------------------------
#------------------------------------------------------------------------------#

# Load the Mosquito Nets Data
nets <- read_rds("data/mosquito_nets.rds")

# Print the data
glimpse(nets)

#------------------------------------------------------------------------------#
#-----------------Causal Graphs for the Treatment Assignment--------------------
#------------------------------------------------------------------------------#

# Construct the DAG to Identify Confounders to Adjust For
mosquito_dag <- dagify(
  malaria_risk ~ net + income + health + temperature,
  net ~ income + health + temperature,
  health ~ income,
  exposure = "net",
  outcome = "malaria_risk",
  coords = list(
    x = c(
      malaria_risk = 7, 
      net = 3, 
      income = 4, 
      health = 5, 
      temperature = 6
      ),
    y = c(
      malaria_risk = 2, 
      net = 2, 
      income = 3, 
      health = 1, 
      temperature = 3
      )
    ),
  labels = c(
    malaria_risk = "Risk of Malaria", 
    net = "Mosquito Net", 
    income = "Income",
    health = "Health", 
    temperature = "Nighttime Temperatures",
    resistance = "Insecticide Resistance"
    )
)

# Turn DAG into a tidy data frame for plotting
mosquito_dag_tidy <- mosquito_dag %>% 
  tidy_dagitty() %>%
  node_status()

# Make things pretty because we aren't savages
mosquito_dag_plot <- ggplot(
  mosquito_dag_tidy, 
  aes(x = x, y = y, xend = xend, yend = yend)
  ) +
  # Add a geom for the graph edges
  geom_dag_edges() +
  # Add a geom for the graph points
  geom_dag_point(aes(color = status)) +
  # Add labels for the graph nodes
  geom_label(
    aes(label = label, fill = status),
    color = "white", 
    fontface = "bold", 
    family = "serif",
    nudge_y = -0.3
    ) +
  # Set the color scheme of the graph nodes
  scale_color_manual(
    values = c(
      exposure = "#5870D0", 
      outcome = "#A00000", 
      latent = "#8830E0"
      ), 
    na.value = "#A868F8"
    ) +
  # Set the fill scheme of the graph nodes
  scale_fill_manual(
    values = c(
      exposure = "#5870D0", 
      outcome = "#A00000", 
      latent = "#8830E0"
      ), 
    na.value = "#A868F8"
    ) +
  # Disable legends
  guides(color = "none", fill = "none") +
  # Apply custom sylistic settings
  map_theme() 

# Render the DAG Plot
print(mosquito_dag_plot)

# Retreive the adjustment set
dagitty::adjustmentSets(mosquito_dag)

#------------------------------------------------------------------------------#
#------------------------Fitting the Design Stage Model-------------------------
#------------------------------------------------------------------------------#

# Note: Since the goal is to predict the assignment of the treatment, in an 
# actual application where there are a wide number of potentially important 
# variables it may make more sense to perform projection predictive variable 
# selection via the projpred package or specify several different treatment
# models and employ a pseudo-bayesian model averaging approach via loo.

# Define a formula for predicting net usage (the treatment)----
treatment_bf <- bf(
  net ~ income + temperature + health,
  family = brmsfamily("bernoulli", "logit"), # Logistic Regression Model
  decomp = "QR" # Applies a QR Decompoisition to the Predictor Matrix
)

# Specify some weakly informative priors for the model parameters----
treatment_priors <- 
  prior("student_t(3, 0, 2.5)", class = "Intercept") +
  prior("normal(0, 2)", class = "b")
  
# Fit the model using brms----
treatment_model <- brm(
  formula = treatment_bf,
  prior = treatment_priors,
  data = nets,
  chains = 8, 
  cores = 8L, 
  iter = 2000, # 1000 warmup/1000 sampling per chain
  opencl = opencl(ids = c(0, 0)), # GPU Accelerated Computation
  seed = 1234, 
  backend = "cmdstanr",
  save_pars = save_pars(all = TRUE),
  save_model = str_c(stan_dir, "Treatment_Model.stan"),
  file = str_c(models_dir, "treatment_model")
)

# Add LOO and Bayes R2 to the Model for the Full Data----
treatment_model <- add_criterion(
  treatment_model,
  criterion = c("loo", "bayes_R2", "loo_R2"),
  cores = 8, # Adjust this based on your system specifications
  file = str_c(models_dir, "treatment_model"),
  seed = 1234
)

# Posterior predictive check for the model
ppc_treatment <- pp_check(treatment_model, ndraws = 1000)

#------------------------------------------------------------------------------#
#------------------------Generating the IPW Weights-----------------------------
#------------------------------------------------------------------------------#

# Note: The `ndraws` argument needs to be equal to at least the total number of 
# iterations (warmup + sampling) that the second stage outcome model will be 
# run for.

# First we need to generate a matrix of expectations from the posterior
pred_probs_chains <- posterior_epred(
  treatment_model,
  cores = 8L,
  seed = 1234
  )

# Then we need to transpose the matrix so that rows correspond to observations
# and each column is a vector of probabilities of length N
ipw_matrix <- t(pred_probs_chains) %>% 
  # Coerce the matrix to a tibble for further pre-processing
  as_tibble() %>% 
  # Add the response vector as the first column
  mutate(net_num = nets$net_num, .before = 1) %>% 
  # Generate the inverse probability weights
  transmute(across(
    starts_with("V"),
    ~ (net_num / .x) + ((1 - net_num) / (1 - .x))
    ))

# Rows are observations and columns are posterior samples
head(ipw_matrix, c(10, 11))

# Write the IPW Matrix to a file
write_rds(ipw_matrix, "data/ipw_matrix.rds")

#------------------------------------------------------------------------------#
#-----------------------Specifying the Outcome Model----------------------------
#------------------------------------------------------------------------------#

# Note: I'm not sure if its possible to specify the outcome model via the 
# method used here in brms. It may be possible to do by specifying a custom 
# family but I'm honestly not sure so for the time being I'm just going to code
# things up directly in Stan. That said, we can use brms to generate the 
# intitial stan code

# Specify the formula for the outcome model----
# outcome_bf <- bf(
#   malaria_risk | weights() ~ net_num,
#   family = brmsfamily("gaussian", "identity") # Normal Regression Model
# )

# Specify some weakly informative priors for the model parameters----
# outcome_priors <- 
#   prior("student_t(3, 0, 2.5)", class = "Intercept") +
#   prior("normal(0, 2.5)", class = "b")

# Generate Stan Code for the Model
#make_stancode(
#  formula = outcome_bf,
#  data = nets,
#  prior = outcome_priors,
#  sample_prior = "no",
#  save_model = str_c(stan_dir, "Outcome_Model.stan"),
#)

# Our goal here is to generate samples based on a different vector
# of the IPW matrix for each iteration. To do this, we need to
# define some C++ function to use in the model. We add the following
# in the functions block of our stan program

"functions {
  // Sets the starting iteration count to 1
  static int itct = 1;
  // Adds 1 to the count each iteration
  inline void add_iter(std::ostream* pstream__){
    itct += 1;
  }
  // Returns the current iteration
  inline int get_iter(std::ostream* pstream__){
    return itct;
  }
}"

# Then we need to declare the dimensions of the ipw_matrix in the 
# data block. We'll remove the existing weights declaration because
# we're going to define the weights as a matrix here

"data {
  int<lower=1> N;  // Total Number of Observations
  vector[N] Y;  // Response Vector
  int<lower=1> K;  // Number of Population-Level Coefficients
  matrix[N, K] X;  // Design Matrix for the Fixed Effects
  int L; // The number of columns in the ipw matrix
  matrix[N, L] IPW; // Declaring the weights matrix
}"

# Then we can add the declaration of the weights matrix in the
# tranformed data block.

"transformed data {
  int Kc = K - 1;
  matrix[N, Kc] Xc;  // Centered version of X without an intercept
  vector[Kc] means_X;  // Column means of X before centering
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
}"

# We don't need to make any changes to the parameters block
"parameters {
  vector[Kc] b;  // Population-Level Effects
  real Intercept;  // Temporary Intercept for Centered Predictors
  real<lower=0> sigma;  // Dispersion Parameter
}"

# In the model block, we'll need to make some changes to the likelihood
# and make the weights a matrix instead of a vector

"model {
  // Likelihood (It might be possible to vectorize this)
  vector[N] mu = Intercept + Xc * b;
  int M = get_iter(); // Get the current iteration
  for (n in 1:N) {
    target += IPW[n, M] * (normal_lpdf(Y[n] | mu[n], sigma));
    }
  // Priors for the Model Parameters
  target += normal_lpdf(b | 0, 2.5);
  target += student_t_lpdf(Intercept | 3, 0, 2.5);
  target += student_t_lpdf(sigma | 3, 0, 14.8) - 1 * student_t_lccdf(0 | 3, 0, 14.8);
}"

# Finally, in generated quantities we need to call the function to update
# the number of iteration counter at each pass.

"generated quantities {
  // Actual Population-Level Intercept
  real b_Intercept = Intercept - dot_product(means_X, b);
  // Update the counter each iteration
  add_iter();
  // You could also calculate the ATE here or whatever
}"

# Then we put it all together in a stan file (not shown) and read back in the 
# modified stan code

# Print the model
str_split(
  read_lines(str_c(stan_dir, "IPW_Outcome_Model.stan")), 
  pattern = ";", 
  simplify = T
  )

# I'm using rstan to fit the outcome model because cmdstanr doesn't
# like my C++ code for some reason

library(rstan)
rstan_options(auto_write = TRUE)

# Compile the stan model
outcome_model <- stan_model(
  model_name = "Outcome IPW Model",
  file = "analyses/stan/IPW_Outcome_Model.stan", 
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
  IPW = as.matrix(ipw_matrix)
)

samp_outcome_model <- sampling(
  outcome_model, 
  data = stan_data_ls, 
  chains = 8, 
  iter = 8000,
  seed = 1234,
  cores = 12,
  sample_file = str_c(models_dir, "outcome-model/IPW_Outcome_Model_Samples"),
  diagnostic_file = str_c(models_dir, "outcome-model/IPW_Outcome_Model_Diag")
)

# Write the stan samples to an object
write_rds(samp_outcome_model, str_c(models_dir, "IPW_Outcome_Model.rds"))
