function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];
% y = (repmat(1:num_labels, m, 1) == y); % M x K
% The below code has the same effect as above
y=[1:num_labels] == y;

% Forward propagation
firstLayerOutput = sigmoid(X * Theta1'); % M x (s_j + 1), s_j=no. of nodes in layer j
secondLayerOutput = sigmoid([ones(m, 1) firstLayerOutput] * Theta2'); % M x K
J = (1/m)*sum(sum((-y.*log(secondLayerOutput) - (1-y).*log(1-secondLayerOutput))));
% Applying regularization term to Cost function J
J = J + (lambda/(2*m))*( sum(sum((Theta1(:, 2:end).^2))) + sum(sum((Theta2(:, 2: end).^2))) );

% Backpropagation

for t = 1:m
   a1 = X(t, :); % 1 x (N+1), this is the input layer nodes containing the training example
   % X has the +1 term from 'ones' matrix already
   z2 = a1 * Theta1'; % 1 x (N+1) * ((N+1) x s_(j+1))

   a2 = [1 sigmoid(z2)]; % 1 x (s_(j+1))


   z3 = a2 * Theta2'; % (1 x (s_j+1)) * ((s_j+1) x K)
   a3 = sigmoid(z3); %  1 x K, K is no. of classifiers

   % calculate delta terms for output layer
   delta_3 = ( a3-y(t, :) ); % 1 x K
   % ( (s_(j+1) x K) * (K x 1) ) .* (1 x s_(j+1))'
   delta_2 = (Theta2' * delta_3').* sigmoidGradient([1 z2])'; 
   % You can either add the 1 to match dimensions, or remove the 1st column
   % of the first multiplicand since the bias term is discarded for the
   % purpose of gradient updating. Since the first delta_2 value is
   % discarded in the accumulation part we chose to match the dimensions
   % for convenience in this case.
   
   % Accumulate gradient from this example
   Theta1_grad = Theta1_grad + (delta_2(2:end) * a1) ; % we skip the bias term
   Theta2_grad = Theta2_grad + delta_3' * a2;
end


Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% % Apply regularization
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m*Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m*Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
