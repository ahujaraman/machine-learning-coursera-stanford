function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% . is used to do this operation on each element
% ./ makes sure that each element in the matrix or vector or scalr is divided individually
e = 2.7183;
exponentialterm  = e.^ (-1*z);
denominator = 1 + exponentialterm;
g = 1 ./ denominator;

% =============================================================

end
