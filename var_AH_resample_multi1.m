function se = var_AH_resample_multi1(beta_hat, data, tau, B)
% var_AH_resample_multi
% Parallel multiplier resampling for multi-type AH model.
%
% This function overwrites the serial version with a parfor-based version.
% Requires Parallel Computing Toolbox.

    beta_hat = beta_hat(:);
    p = length(beta_hat);

    clusters = unique(data.clusterID(:));
    K = length(clusters);

    if nargin < 4 || isempty(B)
        B = 300;
    end

    % Start parallel pool if needed
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local');
    end

    % Preallocate
    beta_star = nan(p, B);

    options = optimset('Display','off', ...
                       'MaxFunEvals', 5e4, ...
                       'MaxIter', 5e4);

    % Fixed quantity outside parfor
    Dk_hat = compute_Dk_AH_multi(beta_hat, data, tau);

    % Reproducible seeds
    baseSeed = 20260421;
    seeds = baseSeed + (1:B);

    parfor b = 1:B
        try
            rng(seeds(b), 'twister');

            G = randn(K,1);
            S = Dk_hat * G;

            obj = @(beta_row) norm(score_AH_multi(beta_row(:), data, tau) - S)^2;

            beta_b_row = fminsearch(obj, beta_hat(:)', options);
            beta_star(:,b) = beta_b_row(:);
        catch ME
            warning('Bootstrap iteration %d failed: %s', b, ME.message);
            beta_star(:,b) = nan(p,1);
        end
    end

    % Drop failed iterations
    ok = all(isfinite(beta_star), 1);
    beta_star = beta_star(:, ok);

    if isempty(beta_star)
        error('All multiplier resampling iterations failed.');
    end

    if size(beta_star,2) < max(30, ceil(0.5*B))
        warning('Only %d/%d bootstrap iterations succeeded.', size(beta_star,2), B);
    end

    se = sqrt(var(beta_star, 0, 2));
end