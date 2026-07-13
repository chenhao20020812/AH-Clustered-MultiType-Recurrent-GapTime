function [U, Uk_list] = score_AH_multi(beta, data, tau)

beta = beta(:);

time      = data.time(:);
delta     = data.delta(:);
Z         = data.Z;
weight    = data.weight(:);
clusterID = data.clusterID(:);
typeID    = data.typeID(:);

R = data.R;

% Linear predictor and transformed gap time
% Truncating eta improves numerical stability during optimization.
eta = Z * beta;
eta = max(min(eta, 50), -50);

Xtilde  = time .* exp(eta);
e_minus = exp(-eta);

p = size(Z,2);
U = zeros(p,1);

clusters = unique(clusterID);
K = length(clusters);
Uk_list = zeros(p,K);

% Map original cluster labels to columns of Uk_list
cluster_index = zeros(length(clusterID),1);
for kk = 1:K
    cluster_index(clusterID == clusters(kk)) = kk;
end

tol = 1e-8;
base = baseline_AH_multi(beta, data, tau);

for r = 1:R
    idx_r = (typeID == r);
    event_times = base(r).t_grid;

    for m = 1:length(event_times)
        t_e = event_times(m);

        % Risk set Y_ijkr(t,beta) = I(X_ijkr exp(beta'Z_ikr) >= t)
        Y = idx_r & (Xtilde >= t_e);
        if ~any(Y)
            continue;
        end

        % Weighted risk-set average Zbar_r(t,beta)
        wYe = weight(Y) .* e_minus(Y);
        denom = sum(wYe);
        if denom <= 0 || ~isfinite(denom)
            continue;
        end

        ZY   = Z(Y,:);
        Zbar = (ZY' * wYe) / denom;

        % Events at transformed time t_e
        idxEv = idx_r & (delta == 1) & (abs(Xtilde - t_e) < tol);
        idList = find(idxEv);

        for s = 1:length(idList)
            id = idList(s);
            contrib = weight(id) * (Z(id,:)' - Zbar);

            U = U + contrib;

            kk = cluster_index(id);
            Uk_list(:,kk) = Uk_list(:,kk) + contrib;
        end
    end
end

end
