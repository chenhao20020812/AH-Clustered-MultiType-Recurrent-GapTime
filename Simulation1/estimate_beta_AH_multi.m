function beta_hat = estimate_beta_AH_multi(data, tau, beta_init)

obj = @(b) norm(score_AH_multi(b, data, tau))^2;

options = optimset('Display','off', ...
                   'MaxFunEvals', 5e4, ...
                   'MaxIter', 5e4);

beta_hat = fminsearch(obj, beta_init(:)', options);
beta_hat = beta_hat(:);
end