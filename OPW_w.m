function [dis,T] = OPW_w(X,Y,a,b,lamda1,lamda2,delta,VERBOSE)
% Compute the Order-Preserving Wasserstein Distance (OPW) for two sequences
% X and Y

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
% a: a N * 1 weight vector for vectors in X, default uniform weights if input []
% b: a M * 1 weight vector for vectors in Y, default uniform weights if input []
% lamda1: the weight of the IDM regularization, default value: 50
% lamda2: the weight of the KL-divergence regularization, default value:
% 0.1
% delta: the parameter of the prior Gaussian distribution, default value: 1
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

if nargin<3 || isempty(lamda1)
    lamda1 = 50;
end

if nargin<4 || isempty(lamda2)
    lamda2 = 0.1;
end

if nargin<5 || isempty(delta)
    delta = 1;
end

if nargin<6 || isempty(VERBOSE)
    VERBOSE = 0;
end

tolerance=.5e-2;
maxIter= 20;
% The maximum number of iterations; with a default small value, the
% tolerance and VERBOSE may not be used;
% Set it to a large value (e.g, 1000 or 10000) to obtain a more precise
% transport;
p_norm=inf;

N = size(X,1);
M = size(Y,1);
dim = size(X,2);
if size(Y,2)~=dim
    disp('The dimensions of instances in the input sequences must be the same!');
end

P = zeros(N,M);
mid_para = sqrt((1/(N^2) + 1/(M^2)));
for i = 1:N
    for j = 1:M
        d = abs(i/N - j/M)/mid_para;
        P(i,j) = exp(-d^2/(2*delta^2))/(delta*sqrt(2*pi));
    end
end

%D = zeros(N,M);
S = zeros(N,M);
for i = 1:N
    for j = 1:M
        %D(i,j) = sum((X(i,:)-Y(j,:)).^2);
        S(i,j) = lamda1/((i/N-j/M)^2+1);
    end
end

D = pdist2(X,Y, 'sqeuclidean');
%D = D/(10^2);
% In cases the instances in sequences are not normalized and/or are very
% high-dimensional, the matrix D can be normalized or scaled as follows:
% D = D/max(max(D));  D = D/(10^2);

K = P.*exp((S - D)./lamda2);
% With some parameters, some entries of K may exceed the maching-precision
% limit; in such cases, you may need to adjust the parameters, and/or
% normalize the input features in sequences or the matrix D; Please see the
% paper for details.
% In practical situations it might be a good idea to do the following:
% K(K<1e-100)=1e-100;

if isempty(a)
    a = ones(N,1)./N;
end

if isempty(b)
    b = ones(M,1)./M;
end

ainvK=bsxfun(@rdivide,K,a);

compt=0;
u=ones(N,1)/N;  

% The Sinkhorn's fixed point iteration
% This part of code is adopted from the code "sinkhornTransport.m" by Marco
% Cuturi; website: http://marcocuturi.net/SI.html
% Relevant paper:
% M. Cuturi,
% Sinkhorn Distances : Lightspeed Computation of Optimal Transport,
% Advances in Neural Information Processing Systems (NIPS) 26, 2013
while compt<maxIter
    u=1./(ainvK*(b./(K'*u)));
    compt=compt+1;    
    % check the stopping criterion every 20 fixed point iterations
    if mod(compt,20)==1 || compt==maxIter   
        % split computations to recover right and left scalings.        
        v=b./(K'*u); 
        u=1./(ainvK*v);
        
        Criterion=norm(sum(abs(v.*(K'*u)-b)),p_norm);
        if Criterion<tolerance || isnan(Criterion), % norm of all || . ||_1 differences between the marginal of the current solution with the actual marginals.
            break;
        end
        
        compt=compt+1;
        if VERBOSE>0
            disp(['Iteration :',num2str(compt),' Criterion: ',num2str(Criterion)]);        
        end                       
    end
end

U = K.*D;
dis=sum(u.*(U*v));
T=bsxfun(@times,v',(bsxfun(@times,u,K)));

end