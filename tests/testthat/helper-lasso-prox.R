# compile/load the C++ function
Rcpp::sourceCpp("../../src/fitLASSOstandardized_prox_Nesterov_c.cpp")

# load the R wrapper
source("../../R/LASSO_wrapper.R")
