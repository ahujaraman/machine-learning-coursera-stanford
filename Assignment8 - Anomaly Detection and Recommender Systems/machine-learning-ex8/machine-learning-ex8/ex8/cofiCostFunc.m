function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
% This elementwise R.* multiplication is for reason, to only incorporrate the R(i,j)=1 values, 
% For only use j who has given rating i
val = R.*( X * Theta') - R.*Y;         
J = 1/2 * sum(sum(dot(val,val)));


% =============================================================


X_grad = val * Theta;
Theta_grad = val' * X;






% =============================================================

% Now we will add the regularized part of component for cimputing cost
% Remember, we are computing Theta, as well as Features X in collaborative learning 
% Thus, we are learning both simlutaneously, so we will have to add both contribution together


% X - num_movies  x num_features matrix of movie features
% We need to sum over all features 1:numfeatures
% For each movie from 1:num_movies
reg_X = (lambda * sum(sum(dot(X,X)))) / 2;



%Theta - num_users  x num_features matrix of user features
reg_Theta = (lambda * sum(sum(dot(Theta,Theta)))) / 2;

J = J + reg_Theta + reg_X;


% =============================================================

reg_X_grad = lambda * X;
reg_Theta_grad = lambda * Theta;

X_grad = X_grad + reg_X_grad;
Theta_grad = Theta_grad + reg_Theta_grad;

grad = [X_grad(:); Theta_grad(:)];
end
