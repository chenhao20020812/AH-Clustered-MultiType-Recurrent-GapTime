function base = baseline_AH_multi(beta, data, tau)

beta = beta(:);

% Basic variables
time   = data.time(:);
delta  = data.delta(:);
Z      = data.Z;
weight = data.weight(:);
typeID = data.typeID(:);

R = data.R;

% Linear predictor and transformed gap time
% Truncating eta improves numerical stability during optimization.
eta = Z * beta;
eta = max(min(eta, 50), -50);

Xtilde  = time .* exp(eta);
e_minus = exp(-eta);

base = struct();
tol = 1e-8;

for r = 1:R
    idx_r = (typeID == r);

    % Event times on the transformed time scale for type r.
    % Only observed events contribute to dNtilde.
    t_grid = unique(Xtilde(idx_r & delta == 1));

    % Restrict the estimating interval to [0, tau].
    if nargin >= 3 && ~isempty(tau)
        t_grid = t_grid(t_grid <= tau);
    end

    M = length(t_grid);

    Lambda0 = zeros(M,1);
    dLambda = zeros(M,1);
    cum = 0;

    for m = 1:M
        t_e = t_grid(m);

        % Risk set Y_ijkr(t,beta) = I(X_ijkr exp(beta'Z_ikr) >= t)
        Y = idx_r & (Xtilde >= t_e);

        if ~any(Y)
            dLam = 0;
        else
            % Denominator: sum w * Y * exp(-beta'Z)
            denom = sum(weight(Y) .* e_minus(Y));

            if denom <= 0 || ~isfinite(denom)
                dLam = 0;
            else
                % Numerator: sum w * dNtilde at transformed event time t_e
                idxEv = idx_r & (delta == 1) & (abs(Xtilde - t_e) < tol);
                num   = sum(weight(idxEv));
                dLam  = num / denom;
            end
        end

        cum = cum + dLam;
        dLambda(m) = dLam;
        Lambda0(m) = cum;
    end

    base(r).t_grid  = t_grid;
    base(r).dLambda = dLambda;
    base(r).Lambda0 = Lambda0;
end

end
