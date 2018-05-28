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
% plainJ, plainGrad are components related to normal logistic regression and then we will add the 
% regularized component part to it, to get the final results
[plainJ,plainGrad] = costFunction(theta, X, y);
% removing the thetazero term from the regularizedComponent
extraComponent = theta(1) * theta(1);
summation = dot(theta,theta);
actualComponent = summation - extraComponent;
regCostComponent = (lambda * actualComponent ) / (2*m);
J =  plainJ + regCostComponent;


% gradient component dont add the component for theta(0), also remember the indexing starts from 1, 
% so accordingly adapt.
% derivative term of regularized part is (lambda/m) * theta(j), 
%exclude for thetazero, i.e theta(1) in octave_config_info

regDerivativeComponent = (lambda/m) * theta;
%Adding first component for thetazero plain 
grad(1) = plainGrad(1);
for iter = 2:size(theta)
  grad(iter) = plainGrad(iter) + regDerivativeComponent(iter);
end

% =============================================================

end
