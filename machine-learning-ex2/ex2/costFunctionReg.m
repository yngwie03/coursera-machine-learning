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




for  i = 1:m

     J =  J + (-1) * y(i,:) * log (sigmoid(X(i,:)*theta)) - (1 - y(i,:)) * log (1 -  sigmoid(X(i,:) * theta));
	       
    for k = 1 : size(theta) 
     grad (k,:)  = grad (k,:)  + (sigmoid(X(i,:)*theta) - y(i,:))* X(i,k);  
	end 
	 
end    
   	

     J = J/m;
	 grad (1,:) = grad (1,:)/m;
	 
	 
for j = 2:	size(theta)

     J = J + lambda * (1/(2*m)) *theta(j,:).^2;
	 
	 grad (j,:) = (grad (j,:) + lambda * theta(j,:))/m;
	
end 
	 	 


% =============================================================

end
