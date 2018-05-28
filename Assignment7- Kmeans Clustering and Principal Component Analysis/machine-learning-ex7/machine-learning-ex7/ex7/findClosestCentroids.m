function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
% outer loop is to loop over training examples one by one
% inner loop is to calculate distance of a particular training example from each centroid
% I am storing this intermediate distance into a result matrix having 2 columns,
% First column -> for which centroid k, 2nd column for the distance 
% Then sort on based of distance 
% And pick up the first centroid (1,1) cell and assign it to that particular training example 


for i= 1: size(X,1)
  cur_X = X(i,:);
  % Making it a column vector
  cur_X = cur_X(:);
  result_matrix = zeros(K,2);
  for k = 1:K
    cur_centroid = centroids(k,:);
    % making it a column vector
    cur_centroid = cur_centroid(:);
    % Distance (x1-x2) 
    numerator = cur_X - cur_centroid;
    dist = sum(dot(numerator,numerator));
    result_matrix(k,:) = [k,dist];
  end 
  % Sort on based of distance and pick first cell of centroid 
  result_matrix = sortrows(result_matrix,2);
  idx(i) = result_matrix(1,1);
end


% =============================================================

end

