function base = baseline_AH_multi(beta, data, tau)

beta = beta(:);

time   = data.time(:);
delta  = data.delta(:);
Z      = data.Z;
weight = data.weight(:);
typeID = data.typeID(:);

R = data.R;

eta     = Z * beta;
Xtilde  = time .* exp(eta);
e_minus = exp(-eta);

base = struct();
tol = 1e-12;

for r = 1:R
    idx_r = (typeID == r);

    % 该类型的事件时间点（只看 delta=1）
    t_grid = unique(Xtilde(idx_r & delta==1));
    M = length(t_grid);

    Lambda0 = zeros(M,1);
    dLambda = zeros(M,1);
    cum = 0;

    for m = 1:M
        t_e = t_grid(m);

        Y = idx_r & (Xtilde >= t_e);
        if ~any(Y)
            dLam = 0;
        else
            denom = sum(weight(Y) .* e_minus(Y));
            if denom <= 0
                dLam = 0;
            else
                idxEv = idx_r & (delta==1) & (abs(Xtilde - t_e) < tol);
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