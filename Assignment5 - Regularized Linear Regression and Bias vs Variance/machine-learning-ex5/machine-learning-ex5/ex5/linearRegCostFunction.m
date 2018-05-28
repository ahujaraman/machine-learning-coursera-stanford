function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



dividend = 2 * m;
prediction = X * theta;
Jtheta = prediction - y;
J = dot(Jtheta,Jtheta) / dividend;


% removing the thetazero term from the regularizedComponent
extraComponent = theta(1) * theta(1);
summation = dot(theta,theta);
actualComponent = summation - extraComponent;
regCostComponent = (lambda * actualComponent ) / (2*m);
J =  J + regCostComponent;





h = prediction;
% Now we calculate gradient 
plainGrad = (X' * ( h - y) )/m;

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







% =========================================================================

grad = grad(:);

end
