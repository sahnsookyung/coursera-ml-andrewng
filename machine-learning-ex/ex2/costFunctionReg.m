function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



% The cost function below utilizes vectorization and does the summation
% The desired cost J is a 1x1 matrix
% y = m x 1; y' = 1 x m
% X = m x n; 
% theta = n x 1
% y' * X*theta = 1 x m * m x 1
% theta(2:end) = 1 x (n-1)

% 
J = (1/m)*((-y' * log(sigmoid( X * theta )))-((1-y)' * log(1-sigmoid(X*theta)))) + ...
(lambda/(2*m)) * theta(2:end)'*theta(2:end);

% grad should have dimensions of theta
% Once again, we use vectorization to calculate the sum.
% theta = n x 1
% X = m x n
% (X*theta-y) = m x 1; (X*theta-y)' = 1 x m
% (1/m)* (sigmoid(X*theta)-y)' * X(:, 2:end) = 1 x (n-1)
% theta(2:end)' = 1 x (n-1)
grad(1) = (1/m) * (sigmoid(X*theta)-y)' * X(:, 1);
grad(2:end) = (1/m) * (sigmoid(X*theta)-y)' * X(:, 2:end) + (lambda/m)*theta(2:end)';



% =============================================================

end
