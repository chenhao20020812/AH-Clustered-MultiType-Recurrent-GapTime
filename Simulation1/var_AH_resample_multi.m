function se = var_AH_resample_multi(beta_hat, data, tau, B)

beta_hat = beta_hat(:);
p = length(beta_hat);

clusters = unique(data.clusterID(:));
K = length(clusters);

beta_star = zeros(p, B);

options = optimset('Display','off', ...
                   'MaxFunEvals', 5e4, ...
                   'MaxIter', 5e4);

Dk_hat = compute_Dk_AH_multi(beta_hat, data, tau);

for b = 1:B
    G = randn(K,1);
    S = Dk_hat * G;

    obj = @(beta_row) norm(score_AH_multi(beta_row(:), data, tau) - S)^2;

    beta_b_row = fminsearch(obj, beta_hat(:)', options);
    beta_star(:,b) = beta_b_row(:);
end

se = sqrt(var(beta_star, 0, 2));
end