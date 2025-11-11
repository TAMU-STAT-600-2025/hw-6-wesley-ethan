#' Multi-class Logistic Regression 
#'
#' @param X n x p training data, 1st column should be 1s to account for intercept
#' @param y a vector of size n of class labels, from 0 to K-1
#' @param numIter number of FIXED iterations of the algorithm, default value is 50
#' @param eta learning rate, default value is 0.1
#' @param lambda ridge parameter, default value is 1
#' @param beta_init (optional) initial starting values of beta for the algorithm, should be p x K matrix 
#'
#'  @return A list containing:
#' \describe{
#'   \item{objective}{Vector of objective function values at each iteration, including initial.}
#' }
#' 
#' @export
#'
#' @examples
#' # Give example
#' set.seed(123)
#' n <- 250 
#' p <- 3 
#' 
#' X <- cbind(1, matrix(rnorm(n * (p-1)), n, p-1))
#' 
#' beta_true <- matrix(c(1.5, -1, 0.5, -1, 1, -0.5, 0, 0, 0.5), nrow=p, byrow=TRUE)
#' 
#' softmax <- function(z) exp(z)/rowSums(exp(z))
#' 
#' probs <- softmax(X %*% beta_true)
#' 
#' y <- apply(probs, 1, function(pr) sample(0:(K-1), 1, prob=pr))
#' 
#' beta_init <- matrix(0, nrow=p, ncol=K)
#' 
#' res_C <- LRMultiClass(X, y, beta_init, numIter=20, eta=0.5, lambda=0.1)
#' 
#' 
#' 
LRMultiClass <- function(X, y, beta_init = NULL, numIter = 50, eta = 0.1, lambda = 1){
  
  # Compatibility checks from HW3 and initialization of beta_init
  n = nrow(X)
  p = ncol(X)
  K = length(unique(y))  # Num classes
  
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  if(!(all(X[, 1] == 1))){
    stop("First column of X not equal to 1's")
  }
  
  # Check for compatibility of dimensions between X and Y
  if(nrow(X) != length(y)){
    stop("Number of rows in X not equal to length of Y")
  }
  
  # Check eta is positive
  if(eta <= 0){
    stop("eta must be > 0")
  }
  
  # Check lambda is non-negative
  if(lambda < 0){
    stop("lambda must be >= 0")
  }
  
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if(is.null(beta_init)){
    beta = matrix(0, nrow = p, ncol = K)
  } else { 
    # beta dim check
    if(nrow(beta_init) != p){
      stop(paste0("beta_init must have ", p, " rows"))
    }
    if(ncol(beta_init) != K){
      stop(paste0("beta_init must have ", K, " cols"))
    }
    
    beta = beta_init 
  }
  
  # Call C++ LRMultiClass_c function to implement the algorithm
  out = LRMultiClass_c(X, y, beta, numIter, eta, lambda)
  
  # Return the class assignments
  return(out)
}
