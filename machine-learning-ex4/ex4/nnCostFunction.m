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
m = size(X, 1); %5000
         
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

%------------------------------------- part 1 ------------------------------

% X:5000 x 400, Theta1 : 25 x 401, Theta2: 10 x 26, y:5000 x 1

% input_layer_size : 400, hidden_layer_size:25

%method 1
%yv: 5000 x10
yv = zeros(m, num_labels);
for i = 1:m
  yv(i, y(i)) = 1;
end

%method 2
%y_matrix = eye(num_labels)(y,:);  % works for Octave

% Add ones to the X data matrix
 X= [ones(m, 1) X]; %5000 x 401 

 z2 = X * Theta1'; % 5000 x 25

 a2 = sigmoid(z2);
 
 a2 = [ones(m, 1) a2]; %5000 x 26 

 z3 = a2 * Theta2'; % 5000 x 10;

 a3 = sigmoid(z3); %5000 x 10


for i = 1:m

for j = 1: num_labels

    J = J  +  (-1)* yv(i,j) * log(a3(i,j)) - (1 - yv(i,j)) * log (1 - a3(i,j)); 
				   
end

end



%25 x 401
rTheta1 = 0;

for i = 1: size(Theta1,1)  %25

	for j = 2: size(Theta1,2)  %401

    rTheta1 = rTheta1 + Theta1(i,j).^2;

    end

 end   

% 10 x 26
rTheta2 = 0;

for i = 1: size(Theta2,1)  %10

	for j = 2: size(Theta2,2)  %26

    rTheta2 = rTheta2 + Theta2(i,j).^2;

    end

 end   


 J = 1/m * (J  + (lambda/2)  * (rTheta1 + rTheta2));

%----------------------------part 1 end ----------------------------------


%----------------------------- part 2-------------------------------------

 %5000 x 10
 d3 = zeros(size(yv));
 
 %5000 x 25
 z2P = sigmoidGradient(z2);

 %5000 x 10
 d3 = a3 - yv;

 %5000 x 25    5000 x 10 10 x 25   element wise  5000 x 25 
 d2 = ( d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

 % 25 x 401    25 x5000 5000 x 401(including bias) 
 Delta1 =  d2' * X;
 %10 x 26   10 x 5000 5000 x 26
 Delta2 = d3' * a2;

 % regulation

 Theta1_temp = Theta1;
 Theta1_temp(:,1) = 0;

 Theta2_temp = Theta2;
 Theta2_temp(:,1) = 0;

 Delta1 = Delta1 + lambda .* Theta1_temp;
 Delta2 = Delta2 + lambda .* Theta2_temp;

 Theta1_grad = Delta1/m ; 
 Theta2_grad = Delta2/m ;

  
  


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
