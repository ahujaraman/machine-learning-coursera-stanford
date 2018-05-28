function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% Outer loop is to iterate over K centroids
% inner loop  indentifies which examples(datapoints) belong to that particular cluster
% idx(i) == k
% Then we calculate mean and assign the new centroid of that cluster


for k =1:K
  cur_centroid = zeros(1,n);
  cluster_points = 0;
  
  for i = 1:size(X,1)
      if idx(i) == k
          cluster_points = cluster_points + 1;
          cur_centroid = cur_centroid + X(i,:);
      end    
  end
  
  cur_centroid = cur_centroid / cluster_points;
  centroids(k,:) = cur_centroid; 
end 




% =============================================================


end

