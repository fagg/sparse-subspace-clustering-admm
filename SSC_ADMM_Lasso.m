% Forms graph for sparse subspace clustering for clean data,
% according to Elhamifar and Vidal (http://arxiv.org/pdf/1203.1005v3.pdf).
%
%  arg min_W,C,E |C|_1
%              s.t. C = W
%                   X = X*W
%                   diag(W) = 0;
%
%   X -> data matrix (MxN)
%   W -> graph matrix (NxN)
%
% Ashton Fagg, ashton@fagg.id.au

function W = SSC_ADMM_Outlier(X, varargin)
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
    C = zeros(N,N);
    W = zeros(N,N);
    V = zeros(N,N);
    U = zeros(M,N);
    
    lambda = estimate_lambda(X);
    rho = estimate_rho(X);
    alpha = rho/2;
    
    for iteration = 1:maxIterations
        W = pinv(rho*X'*X+alpha*eye(N,N))*(rho*(X'*X + X'*U) + alpha*(C+V));
        W = W - diag(diag(W));
        C = soft_threshold(W+V, 2/alpha);
        C = C - diag(diag(C));
        U = U + (X-X*W);
        V = V + (C-W);

        if (printFlag)
            print_status(iteration, C, W, X, lambda);
        end
        
        if(norm(X-X*W,'fro')<1e-4 && norm(C-W,'fro')<1e-4)
            break;
        else
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
end

function print_status(it, C, W, X, lambda)
    disp(['Iteration: ', num2str(it)]);
    obj = norm(W,1);
    disp(['||W||_1  = ', num2str(obj)]);
    disp(['||X-XW||_F = ', num2str(norm(X-X*W,'fro'))]);
    disp(['||C-W||_F = ', num2str(norm(C-W,'fro'))]);
    disp('-------------------------------');
end
