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
    
Theta1_reg_component = Theta1;
Theta2_reg_component = Theta2;
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

% Add ones to the X data matrix
X = [ones(m, 1) X];

  % The matrix X contains the examples in rows
  % Size of X = 5000 * 401
  % Here, we have 5000 training examples
  % For each example we have 400 input pixels i.e. it is an image of 20 * 20
  % Which is stored as a row vector, i.e (i) th reperesents the pixels of (i) th training picture
  % We add one more node to each example as zero-th node, thus we have 401 columns now
  % First column for each row represents the bias unit = 1
  % activation1 = input layer
    activation1 = X;
   
    % For calculating activation of second layer, z = theta1 * activation1 
    % activation2 = g(z), sigmoid(z)
    % Theta1 : 25 * 401
    % hidden layer has : 25 units 
    % Each node in the input layer has 25 weights associated for each the hidden layer unit
    % As, we have 401 input nodes for each picture example, thus Theta1(weight_vector) = 25 * 401
    % Take Theta1' = 401 * 25
    % z2 = (5000 * 401) * (401 * 25) = ( 5000 * 25 ) i.e activation function for 25 hidden layer units
    z2 =  activation1 * Theta1';
    activation2 = sigmoid(z2);
    
    % Similarly calculate for final layer, remember this can cosmetically similar to h(theta) for logistic regression
    % If we imagine activation of previous layer as input set and then multiply to weights (Theta) in this case of nueral nets


    % Theta2 : =  10 *26 ( Output layer has to be divided into 10 classes)
    % 26 : = As , we add bias unit node
    % Theta2' = 26 *10
    % activation2 = 5000 * 26 (after adding bias unit)
    % z3 := (5000*26) * (26 *10) = 5000 * 10
    activation2 = [ones(m, 1) activation2];
    z3 = activation2 * Theta2';
    activation3 = sigmoid(z3);

    % Final layer activation is hypothesis output
    h = activation3;
    
    % Thus, we get h_theta_x := 5000 *10, for each row i.e training picture we have 10 lables of weight
    % Of same dimension we create y: 5000 *10
    % y(k) - vectors containing only values 0 or 1 (5000 * 10)
      y_new = zeros(m,num_labels); 
      
    % In this we give value 1 for correct clssification all other zeros, for each example in 5000 set
    for i=1:m
        y_new(i,y(i))=1;
    end

    % we apply the logistic regression formula, to dot multiply each element to its corresponding hypothesis wieght 
    % We iterate over k-classes for each i =1:m examples
    % Iterate, here dont take literal meaning of for loop, mathematically dot product will take care of it
   
    J = (1/m) * sum ( sum ( (-y_new) .* log(h) - (1-y_new) .* log(1-h) ));
  
    % Regularization component,, I have removed the first column from both matrix.
    % Because we dont want bias unit to contribute to regularization
    Theta1_new = Theta1(:, [ 2:end]);
    Theta2_new = Theta2(:, [ 2:end]);
    component1 = sum(dot(Theta1_new,Theta1_new));
    component2 = sum(dot(Theta2_new,Theta2_new));
    reg_component = lambda * ( component1+ component2) / (2*m);
 
    % Adding the regularized component. 
    J = J + reg_component;


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

size(X(2,:)')

for t=1:m
  
  
    % As we have already added ones, dimension of activation1 := 401 * 1
    activation1 = X(t,:)';
  
    % theta1 : = 25 * 401 
    % z2: = ( 25 * 401) * (401 * 1) = (25 *1)
    z2 =   Theta1 * activation1 ;
    activation2 = sigmoid(z2);
    activation2 = [1;activation2];
    % activation2 : = 26 *1 
    % theta2 : = 10 * 26
    % z3 = (10*26) * (26 *1) = (10*1)
    z3 =  Theta2 * activation2 ;
    activation3 = sigmoid(z3);
    
    % we have column vector of, 10*1
    y_cur  = [1:num_labels]' == y(t);
    
    delta3 = activation3 - y_cur;

    % we have to exclude bias to calculate delta of previous layer , her ;layer2
    % delta3 = (10*1)
    % (Theta2 excluidng the bais unit)' = 25 * 10 
    
  
    % delta2 = (25*10) * (10*1) = (25*1) 
    delta2 = (Theta2(:,2:end)' * delta3 ) .* sigmoidGradient(z2);
    % Theta1_grad : = (25*1) * (1*401) = 25* 401
    Theta1_grad = Theta1_grad + delta2 * activation1';
    Theta2_grad = Theta2_grad + delta3 * activation2';
end 


Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;



% Regularization by adding the reg term, excluding the condition of bias term
% Simple way of excluding bias units in contribution to regularization term is 
% make Theta's first column to Zero
 Theta1(:,1) =0;
 Theta2(:,1) =0;

Theta1_grad = Theta1_grad + (lambda/m) * Theta1;
Theta2_grad = Theta2_grad + (lambda/m) * Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
