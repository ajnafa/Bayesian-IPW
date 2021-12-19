// Simple Example of a Bayesian Inverse Probability Weighted Outcome Model
functions {
  void add_iter();
  int get_iter();
}

data {
  int<lower=1> N;  // Total Number of Observations
  vector[N] Y;  // Response Vector
  int<lower=1> K;  // Number of Population-Level Coefficients
  matrix[N, K] X;  // Design Matrix for the Fixed Effects
  int L; // The number of columns in the ipw matrix
  matrix[N, L] IPW; // Declaring the weights matrix
}

transformed data {
  int Kc = K - 1;
  matrix[N, Kc] Xc;  // Centered version of X without an intercept
  vector[Kc] means_X;  // Column means of X before centering
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
}

parameters {
  vector[Kc] b;  // Population-Level Effects
  real Intercept;  // Temporary Intercept for Centered Predictors
  real<lower=0> sigma;  // Dispersion Parameter
}

model {
  // Likelihood (It might be possible to vectorize this)
  vector[N] mu = Intercept + Xc * b;
  int M = get_iter();
  for (n in 1:N) {
    target += IPW[n, M] * (normal_lpdf(Y[n] | mu[n], sigma));
  }
  // Priors for the Model Parameters
  target += normal_lpdf(b | 0, 2.5);
  target += student_t_lpdf(Intercept | 3, 0, 2.5);
  target += student_t_lpdf(sigma | 3, 0, 14.8) - 1 * student_t_lccdf(0 | 3, 0, 14.8);
}

generated quantities {
  // Actual Population-Level Intercept
  real b_Intercept = Intercept - dot_product(means_X, b);
  // Update the counter each iteration
  add_iter();
}
