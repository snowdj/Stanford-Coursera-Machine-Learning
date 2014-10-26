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

% Part 1

   % Feed forward through neural network
      % Add ones to the X data matrix to get A^(1)
      A1 = [ones(m, 1) X];

      % calculations for first hidden layer
      Z2 = A1 * Theta1';
      A2 = sigmoid(Z2);
      %Add ones to the A^(2) matrix
      A2 = [ones(m, 1) A2];

      % Calculations for the output layer
      Z3 = A2 * Theta2';
      A3 = sigmoid(Z3);

      % Compute the log of hypotheses
      h1 = log(A3);
      h2 = log(1-A3);
  
   % Compute the unregularized cost function
      % Create identity matrix to the size of num_labels
      I = eye(num_labels);
      
      % Pick the appropriate standard unit vector that corresponds to the y value
      % and compute a vector for the inner sum
      
      % Create logical matrix
      Y_mat = zeros(size(A3));
      for i=1:m
         Y_mat(i, :) = I(y(i), :);
      end;
      J_unreg = 0;
      for i=1:m
        J_unreg = J_unreg + h1(i,:)*Y_mat(i, :)' + h2(i,:)*(1-Y_mat(i, :)'); 
      end;

    % Regularize the cost function
       Theta1_sq = Theta1(:,2:end).^2;
       Theta2_sq = Theta2(:,2:end).^2; 
       J_reg = lambda/(2*m)*(sum(sum(Theta1_sq)) ...
             + sum(sum(Theta2_sq)));
       J = -1/m*J_unreg + J_reg;
       
     % Backpropagation
     delta3 = A3 - Y_mat;
     delta2 = delta3*Theta2(:,2:end).*sigmoidGradient(Z2);
     Delta2 = A2'*delta3;
     Delta2 = Delta2';
     Delta1 = delta2'*A1;
     
     Theta1_grad = 1/m*Delta1;
     Theta2_grad = 1/m*Delta2;
     
     % Gradient regularizaton
     Theta2 = [zeros(num_labels, 1) Theta2(:, 2:end)];
     Theta1 = [zeros(hidden_layer_size, 1) Theta1(:, 2:end)];
     
     Theta1_grad = Theta1_grad + lambda/m*Theta1;
     Theta2_grad = Theta2_grad + lambda/m*Theta2;
 
 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
