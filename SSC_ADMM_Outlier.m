% Forms graph for sparse subspace clustering with outlier considerations,
% according to Elhamifar and Vidal (http://arxiv.org/pdf/1203.1005v3.pdf).
%
%  arg min_W,C,E |C|_1 + lambda * |E|_1
%              s.t. C = W
%                   X = X*W + E
%                   diag(W) = 0;
%
%   X -> data matrix (MxN)
%   W -> graph matrix (NxN)
%   E -> outlier matrix (MxN)
%
% Ashton Fagg, ashton@fagg.id.au

function [W,E] = SSC_ADMM_Outlier(X, varargin)
    if (nargin == 1)
       maxIterations = 10000;
       printFlag = false;
    elseif (nargin == 2)
       maxIterations = varargin{1};
       printFlag = false;
    elseif (nargin == 3)
       maxIterations = varargin{1};
       printFlag = varargin{2};
    else
        error('ssc:admm:huh', 'Unknown configuration.');
    end

    [M,N] = size(X);
    E = zeros(M,N);
    C = zeros(N,N);
    W = zeros(N,N);
    V = zeros(N,N);
    U = zeros(M,N);
    Wprev = randn(N,N);
    
    lambda = estimate_lambda(X);
    rho = estimate_rho(X);
    alpha = rho*5;
    
    for iteration = 1:maxIterations
        W = pinv(rho*X'*X+alpha*eye(N,N))*(rho*(X'*X - X'*E + X'*U) + alpha*(C+V));
        W = W - diag(diag(W));
        E = soft_threshold(X-X*W+U,2*lambda/rho);
        C = soft_threshold(W+V, 2/alpha);
        C = C - diag(diag(C));
        U = U + (X-X*W-E);
        V = V + (C-W);

        if (printFlag)
            print_status(iteration, C, W, E, X, lambda,Wprev);
        end
        
        if(iteration>20 && norm(W-Wprev,'fro')<1e-5)
            break;
        else
            Wprev = W;
            continue;  
        end
        
    end
    W = normalise_final_W(W);
end

function x = soft_threshold(b, m)
  x = sign(b)  .* max( abs(b)-m, 0 );
end

function lambda = estimate_lambda(X)
    M = size(X,1);
    lambda = 1/sqrt(M);
end

function rho = estimate_rho(X)
    rho = 1/(2*norm(X));
end

function finalW = normalise_final_W(W)
    finalW = zeros(size(W));
    N = size(W,2);
    for i = 1:N
       Wi = W(:,i);
       Wi = Wi / norm(Wi, Inf);
       finalW(:,i) = Wi;
    end
    finalW = abs(finalW) + abs(finalW');
end

function print_status(it, C, W, E, X, lambda,Wprev)
    %disp(['||W||_1 ', num2str(norm(W,1))]);
    %disp(['||C||_1' , num2str(norm(C,1))]);
    %disp(['||E||_1' , num2str(norm(E,1))]);
    disp(['Iteration: ', num2str(it)]);
    obj = norm(W,1) + lambda*norm(E,1);
    disp(['||W||_1 + lambda*||E||_1 = ', num2str(obj)]);
    disp(['||X-XW-E||_F = ', num2str(norm(X-X*W-E,'fro'))]);
    disp(['||C-W||_F = ', num2str(norm(C-W,'fro'))]);
    disp(['||W_k+1 - W_k||_F = ' num2str(norm(W-Wprev,'fro'))]);
    disp('-------------------------------');
end