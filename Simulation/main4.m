clear; clc;
tic;

% ============================================================
% Two-type simulation: Type 1 + Type 3
%
% Internal coding:
%   typeID = 1  -> original type 1: lambda01(t) = t
%   typeID = 2  -> original type 3: lambda03(t) = 1/(1+t)
%
% This is still R = 2 for estimation functions.
% ============================================================

tau   = 4;
beta0 = [1; 0.5];
R     = 2;

% ------------------------------------------------------------
% Simulation settings
% ------------------------------------------------------------
nClusters_set = [50];

nRep   = 100;
B_boot = 50;

rho_triplets = [
    0.1 0.1 0.1;
    0.1 0.2 0.3;
    0.3 0.3 0.3
];

% censoring setting
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
        avgS_res = zeros(nRep,1);

        fprintf('\n====================================================\n');
        fprintf('Two-type simulation: original type1 + original type3\n');
        fprintf('n=%d, rho=(%.1f, %.1f, %.1f), tau=%.1f\n', ...
            nClusters, rho1, rho2, rho3, tau);
        fprintf('nu1=%.1f, nu2=%.1f, beta0=(%.1f, %.1f)\n', ...
            nu1, nu2, beta0(1), beta0(2));
        fprintf('Internal typeID: 1=original type1, 2=original type3\n');
        fprintf('nRep=%d, B_boot=%d\n', nRep, B_boot);
        fprintf('====================================================\n');

        parfor rep = 1:nRep

            % ------------------------------------------------
            % Generate data using original type1 and type3
            % ------------------------------------------------
            data = simulate_dataset_AH_multi2(nClusters, ...
                rho1, rho2, rho3, beta0, tau, nu1, nu2);

            % ------------------------------------------------
            % Estimate beta
            % ------------------------------------------------
            beta_hat = estimate_beta_AH_multi(data, tau, beta0);

            % ------------------------------------------------
            % Multiplier resampling SE
            % ------------------------------------------------
            se_hat = var_AH_resample_multi(beta_hat, data, tau, B_boot);

            CI_low  = beta_hat - 1.96 * se_hat;
            CI_high = beta_hat + 1.96 * se_hat;

            cp_tmp = [
                (CI_low(1) <= beta0(1) && beta0(1) <= CI_high(1)), ...
                (CI_low(2) <= beta0(2) && beta0(2) <= CI_high(2))
            ];

            beta_res(rep,:) = beta_hat(:).';
            se_res(rep,:)   = se_hat(:).';
            cp_res(rep,:)   = cp_tmp;

            % Average observed gap number per subject-type
            G = findgroups(data.subjectID, data.typeID);
            avgS_res(rep) = mean(splitapply(@numel, data.time, G));
        end

        % ====================================================
        % Summary
        % ====================================================
        summary.bias  = mean(beta_res - beta0.', 1);
        summary.seEmp = std(beta_res, 0, 1);
        summary.ese   = mean(se_res, 1);
        summary.cp    = mean(cp_res, 1);
        summary.avgS  = mean(avgS_res);

        fprintf('\nAverage observed gap number = %.4f\n', summary.avgS);

        fprintf('beta1: Bias=% .4f, SE=% .4f, ESE=% .4f, CP=%.3f\n', ...
            summary.bias(1), summary.seEmp(1), summary.ese(1), summary.cp(1));

        fprintf('beta2: Bias=% .4f, SE=% .4f, ESE=% .4f, CP=%.3f\n', ...
            summary.bias(2), summary.seEmp(2), summary.ese(2), summary.cp(2));

        % ====================================================
        % Save into Results
        % ====================================================
        key = sprintf('type13_rho_%02d_%02d_%02d_n_%d', ...
            round(100*rho1), round(100*rho2), round(100*rho3), nClusters);

        Results.(key) = summary;
        Results.(key).beta0      = beta0.';
        Results.(key).rho1       = rho1;
        Results.(key).rho2       = rho2;
        Results.(key).rho3       = rho3;
        Results.(key).nClusters  = nClusters;
        Results.(key).nRep       = nRep;
        Results.(key).B_boot     = B_boot;
        Results.(key).tau        = tau;
        Results.(key).R          = R;
        Results.(key).nu1        = nu1;
        Results.(key).nu2        = nu2;
        Results.(key).typeSet    = [1 3];

        Results.(key).beta_res   = beta_res;
        Results.(key).se_res     = se_res;
        Results.(key).cp_res     = cp_res;
        Results.(key).avgS_res   = avgS_res;

    end
end

runTime = toc;

save('AH_multi_type13_two_type_simulation.mat', 'Results', 'runTime');

fprintf('\n====================================================\n');
fprintf('Finished. Time = %.2f minutes\n', runTime/60);
fprintf('Results saved to AH_multi_type13_two_type_simulation.mat\n');
fprintf('====================================================\n');
