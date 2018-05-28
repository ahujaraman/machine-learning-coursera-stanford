function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



% Add ones to the X data matrix
X = [ones(m, 1) X];

% activation azero, which will be input layer as it is 
activation1 = X;

% For calculating activation of second layer, z = theta1 * activation1 
% activation2 = g(z), sigmoid(z)
z2 = activation1 * Theta1';
activation2 = sigmoid(z2);

% Similarly calculate for final layer, remember this can cosmetically similar to h(theta) for logistic regression
% If we imagine activation of previous layer as input set and then multiply to weights (Theta) in this case of nueral nets


activation2 = [ones(m, 1) activation2];
z3 = activation2 * Theta2';
activation3 = sigmoid(z3);

% Final layer activation is hypothesis output

h = activation3;


% Time to predict 
[values,column_indices] = max(h,[],2);
p = column_indices; 

% =========================================================================


end
