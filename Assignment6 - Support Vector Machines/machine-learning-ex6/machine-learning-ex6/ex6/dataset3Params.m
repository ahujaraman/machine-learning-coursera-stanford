function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
range_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
n = length(range_vec);
result_matrix = zeros(n*n,3);
error_number = 1;
for i = 1:length(range_vec)
  Cur_C = range_vec(i);
  for j = 1:length(range_vec)
    Cur_sigma = range_vec(j);
    model= svmTrain(X, y, Cur_C, @(x1, x2) gaussianKernel(x1, x2, Cur_sigma));
    predictions = svmPredict(model, Xval);
    Cur_error = mean(double(predictions ~= yval));
    
  result_matrix(error_number,:) = [Cur_C,Cur_sigma,Cur_error];
  error_number = error_number + 1;
  end
  
end 



result_matrix = sortrows(result_matrix,3);

C = result_matrix(1,1);
sigma = result_matrix(1,2);

% =========================================================================

end
