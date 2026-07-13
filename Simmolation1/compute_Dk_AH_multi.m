function Dk = compute_Dk_AH_multi(beta, data, tau)

beta = beta(:);

time      = data.time(:);
delta     = data.delta(:);
Z         = data.Z;
clusterID = data.clusterID(:);
subjectID = data.subjectID(:);
typeID    = data.typeID(:);

R = data.R;

eta     = Z * beta;
Xtilde  = time .* exp(eta);
e_minus = exp(-eta);

clusters = unique(clusterID);
K = length(clusters);
p = size(Z,2);
nTotal = length(time);

Dk = zeros(p, K);

cluster_index = zeros(nTotal,1);
nk_vec        = zeros(nTotal,1);
Sikr_vec      = zeros(nTotal,1);

% 计算每个观测对应的 cluster 下标、n_k、S_ikr
for kk = 1:K
    k = clusters(kk);
    idx_k = (clusterID == k);
    cluster_index(idx_k) = kk;

    subjects_k = unique(subjectID(idx_k));
    nk = length(subjects_k);

    for ii = 1:length(subjects_k)
        sid = subjects_k(ii);
        for r = 1:R
            idx_ikr = idx_k & (subjectID == sid) & (typeID == r);
            if any(idx_ikr)
                Sikr = sum(idx_ikr);
                nk_vec(idx_ikr)   = nk;
                Sikr_vec(idx_ikr) = Sikr;
            end
        end
    end
end

base_weight = 1 ./ (nk_vec .* Sikr_vec);

base = baseline_AH_multi(beta, data, tau);
tol = 1e-10;

for r = 1:R
    idx_r = (typeID == r);

    t_grid  = base(r).t_grid;
    dLambda = base(r).dLambda;

    for m = 1:length(t_grid)
        t_m = t_grid(m);
        dLambda_m = dLambda(m);

        at_risk = idx_r & (Xtilde >= t_m);
        if ~any(at_risk)
            continue;
        end

        idx_at = find(at_risk);
        Z_at   = Z(idx_at,:);

        w_e = base_weight(idx_at) .* e_minus(idx_at);
        S0  = sum(w_e);
        if S0 <= 0
            continue;
        end

        S1   = Z_at' * w_e;
        Zbar = S1 / S0;

        is_event_at_t = (delta(idx_at) == 1) & (abs(Xtilde(idx_at) - t_m) < tol);

        % dM = dN~ - Y exp(-eta) dLambda
        dM_vec = double(is_event_at_t) - e_minus(idx_at) * dLambda_m;

        w0_at = base_weight(idx_at);

        for rr = 1:length(idx_at)
            obs = idx_at(rr);
            kk  = cluster_index(obs);

            contrib = (Z(obs,:)' - Zbar) * (w0_at(rr) * dM_vec(rr));
            Dk(:,kk) = Dk(:,kk) + contrib;
        end
    end
end
end