function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% we have a matrix of m training examples and n features 
% X = m * n; Also, mu is mean of each feature, so we sum over each column.
% i.e it has values for all examples for that feature
% For mean, we divide it by / m.

for i=1:n
  mu(i) = sum(X(:,i)) / m;
end




% Variance of each feature
% Imagine column 1 as feature 1 values and it has m rows. As we have m training examples (samples)
% For this feature if we have to calculate sigma2: We will subtract (x1 - mu1)2 for all m examples and divide / m.


for i = 1:n
  val = X(:,i) - mu(i);
  % sum over -> square of difference
  numerator = sum(dot(val,val));
  sigma2(i) = numerator / m;
end 




% =============================================================


end
