function [dis,T] = TCOT(X,Y,lambda,VERBOSE)
% Compute the Temporally Coupled Optimal Transport (TCOT) distance for two
% sequences X and Y

% -------------
% DEPENDENCY:
% -------------
% "sinkhornTransport.m" by Marco Cuturi; website: http://marcocuturi.net/SI.html
% Please download and add the code into the current directory
% Relevant paper:
% M. Cuturi,
% Sinkhorn Distances : Lightspeed Computation of Optimal Transport,
% Advances in Neural Information Processing Systems (NIPS) 26, 2013

% -------------
% INPUT:
% -------------
% X: a N * d matrix, representing the input sequence consists of of N
% d-dimensional vectors, where N is the number of instances (vectors) in X,
% and d is the dimensionality of instances;
% Y: a M * d matrix, representing the input sequence consists of of N
% d-dimensional vectors, , where N is the number of instances (vectors) in
% Y, and d is the dimensionality of instances;
% iterations = total number of iterations
% lamda: the weight of the entropy regularization, default value: 1
% VERBOSE: whether display the iteration status, default value: 0 (not display)

% -------------
% OUTPUT
% -------------
% dis: the OPW distance between X and Y
% T: the learned transport between X and Y, which is a N*M matrix


% -------------
% c : barycenter according to weights
% ADVICE: divide M by median(M) to have a natural scale
% for lambda

% -------------
% Copyright (c) 2017 Bing Su, Gang Hua
% -------------
%
% -------------
% License
% The code can be used for research purposes only.

if nargin<3 || isempty(lambda)
    lambda = 1;
end

if nargin<4 || isempty(VERBOSE)
    VERBOSE = 0;
end

tolerance=.5e-2;
maxIter=100;
% The maximum number of iterations;
% Set it to a large value (e.g, 1000 or 10000) to obtain a more precise
% transport;
p_norm=inf;

N = size(X,1);
M = size(Y,1);
dim = size(X,2);
if size(Y,2)~=dim
    disp('The dimensions of instances in the input sequences must be the same!');
end

D = zeros(N,M);
for i = 1:N
    for j = 1:M
        D(i,j) = sum((X(i,:)-Y(j,:)).^2);
        D(i,j) = D(i,j)*(1+abs(i/N-j/M));
    end
end

% D = pdist2(X,Y, 'sqeuclidean');
% for i = 1:N
%     for j = 1:M
%         D(i,j) = D(i,j)*(1+abs(i/N-j/M));
%     end
% end

% In cases the instances in sequences are not normalized and/or are very
% high-dimensional, the matrix D can be normalized or scaled as follows:
% D = D/max(max(D));  D = D/(10^2);


K=exp(-lambda*D);
% With some parameters, some entries of K may exceed the maching-precision
% limit; in such cases, you may need to adjust the parameters, and/or
% normalize the input features in sequences or the matrix D; Please see the
% paper for details.
% In practical situations it might be a good idea to do the following:
% K(K<1e-100)=1e-100;

U=K.*D;

a = ones(N,1)./N;
b = ones(M,1)./M;

% Call the dependency "sinkhornTransport.m" to solve the matrix scaling
% problem
[dis,lowerEMD,l,m]=sinkhornTransport(a,b,K,U,lambda,[],p_norm,tolerance,maxIter,VERBOSE);
T=bsxfun(@times,m',(bsxfun(@times,l,K))); % this is the optimal transport.

end