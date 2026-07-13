clear; clc;
tic;

tau   = 4;
beta0 = [1; 0.5];
R     = 2;

nClusters_set = [50 100];
nRep   = 1000;
B_boot = 500;   % 调试时可减小，正式建议再加大

rho_triplets = [
    0.1 0.1 0.1;
    0.3 0.3 0.3
];

% 你这里要自己调到“平均 recurrence number 在 3~6 之间”
nu1 = 2;
nu2 = 4;

if isempty(gcp('nocreate'))
    parpool;
end

Results = struct();

for rr = 1:size(rho_triplets,1)
    rho1 = rho_triplets(rr,1);
    rho2 = rho_triplets(rr,2);
    rho3 = rho_triplets(rr,3);

    for ii = 1:length(nClusters_set)
        nClusters = nClusters_set(ii);

        beta_res = zeros(nRep,2);
        se_res   = zeros(nRep,2);
        cp_res   = zeros(nRep,2);

        fprintf('\nn=%d, rho=(%.1f,%.1f,%.1f)\n', ...
            nClusters, rho1, rho2, rho3);

        parfor rep = 1:nRep
            data = simulate_dataset_AH_multi(nClusters, R, ...
                rho1, rho2, rho3, beta0, tau, nu1, nu2);

            beta_hat = estimate_beta_AH_multi(data, tau, beta0);
            se_hat   = var_AH_resample_multi(beta_hat, data, tau, B_boot);

            CI_low  = beta_hat - 1.96 * se_hat;
            CI_high = beta_hat + 1.96 * se_hat;

            cp_tmp = [ ...
                (CI_low(1) <= beta0(1) && beta0(1) <= CI_high(1)), ...
                (CI_low(2) <= beta0(2) && beta0(2) <= CI_high(2)) ...
            ];

            beta_res(rep,:) = beta_hat.';
            se_res(rep,:)   = se_hat.';
            cp_res(rep,:)   = cp_tmp;
        end

        summary.bias  = mean(beta_res - beta0.', 1);
        summary.seEmp = std(beta_res, 0, 1);
        summary.ese   = mean(se_res, 1);
        summary.cp    = mean(cp_res, 1);

        fprintf('beta1: Bias=% .4f, SE=% .4f, ESE=% .4f, CP=%.3f\n', ...
            summary.bias(1), summary.seEmp(1), summary.ese(1), summary.cp(1));
        fprintf('beta2: Bias=% .4f, SE=% .4f, ESE=% .4f, CP=%.3f\n', ...
            summary.bias(2), summary.seEmp(2), summary.ese(2), summary.cp(2));

        key = sprintf('rho_%02d_%02d_%02d_n_%d', ...
            round(100*rho1), round(100*rho2), round(100*rho3), nClusters);

        Results.(key) = summary;
        Results.(key).beta0   = beta0.';
        Results.(key).rho1    = rho1;
        Results.(key).rho2    = rho2;
        Results.(key).rho3    = rho3;
        Results.(key).nRep    = nRep;
        Results.(key).B_boot  = B_boot;
        Results.(key).tau     = tau;
        Results.(key).R       = R;
        Results.(key).nu1     = nu1;
        Results.(key).nu2     = nu2;
    end
end

runTime = toc;
save('AH_multi_simulation.mat', 'Results', 'runTime');
fprintf('\nFinished. Time = %.2f minutes\n', runTime/60);