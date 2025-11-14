test_that("fitLASSO_prox_Nesterov returns correct shapes and fmin", {
  set.seed(123)
  n <- 50
  p <- 10
  
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(runif(4, -1, 1), rep(0, p - 4))
  Y <- as.numeric(X %*% beta_true + rnorm(n, sd = 0.5))
  
  lambda <- 0.2
  
  out <- fitLASSO_prox_Nesterov(X, Y, lambda = lambda)
  
  # beta should be intercept + p coefficients
  expect_type(out$beta, "double")
  expect_length(out$beta, p + 1)
  
  # fmin should be numeric scalar
  expect_type(out$fmin, "double")
  expect_length(out$fmin, 1L)
  
  # Manually recompute objective on original scale and compare
  b0 <- out$beta[1]
  b  <- out$beta[-1]
  resid <- Y - (b0 + as.vector(X %*% b))
  f_manual <- 0.5 / n * sum(resid^2) + lambda * sum(abs(b))
  
  expect_equal(out$fmin, f_manual, tolerance = 1e-6)
})

test_that("larger lambda gives smaller or equal L1 norm of coefficients", {
  set.seed(456)
  n <- 60
  p <- 12
  
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(runif(5, -1, 1), rep(0, p - 5))
  Y <- as.numeric(X %*% beta_true + rnorm(n, sd = 0.7))
  
  lambda_small <- 0.05
  lambda_large <- 0.5
  
  out_small <- fitLASSO_prox_Nesterov(X, Y, lambda = lambda_small)
  out_large <- fitLASSO_prox_Nesterov(X, Y, lambda = lambda_large)
  
  # L1 norm of coefficients (exclude intercept)
  l1_small <- sum(abs(out_small$beta[-1]))
  l1_large <- sum(abs(out_large$beta[-1]))
  
  expect_true(l1_large <= l1_small + 1e-8)
})

test_that("non-null beta_start is accepted and has correct length", {
  set.seed(789)
  n <- 40
  p <- 8
  
  X <- matrix(rnorm(n * p), n, p)
  Y <- rnorm(n)
  lambda <- 0.1
  
  beta_start <- rep(0.5, p)
  
  out <- fitLASSO_prox_Nesterov(X, Y, lambda = lambda, beta_start = beta_start)
  
  expect_length(out$beta, p + 1)
  expect_true(is.finite(out$fmin))
})

