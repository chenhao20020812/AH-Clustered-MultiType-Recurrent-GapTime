function data = simulate_dataset_AH_multi2(nClusters, rho1, rho2, rho3, beta0, tau, nu1, nu2)
% simulate_dataset_AH_multi_type13
%
% Generate clustered two-type recurrent gap-time data using:
%
%   internal typeID = 1  -> original type 1:
%       lambda01(t) = t
%       T0 = exp(eta) * T^2 / 2
%       T  = sqrt(2*T0*exp(-eta))
%
%   internal typeID = 2  -> original type 3:
%       lambda03(t) = 1/(1+t)
%       T0 = exp(-eta) * log(1 + exp(eta)*T)
%       T  = exp(-eta) * ( exp(exp(eta)*T0) - 1 )
%
% The estimation functions should still use R = 2.
%
% Inputs:
%   nClusters : number of clusters
%   rho1      : cluster-level heterogeneity variance
%   rho2      : subject-level heterogeneity variance
%   rho3      : type-specific heterogeneity variance
%   beta0     : true beta, p x 1
%   tau       : truncation time
%   nu1, nu2  : censoring time Uniform(nu1,nu2), truncated by tau
%
% Output:
%   data      : structure compatible with estimate_beta_AH_multi,
%               score_AH_multi and var_AH_resample_multi.

    beta0 = beta0(:);
    p = length(beta0);
    R = 2;

    rho4 = 1 - rho1 - rho2 - rho3;
    if rho4 < 0
        error('rho1 + rho2 + rho3 must be <= 1.');
    end

    % Storage
    time_list      = [];
    delta_list     = [];
    Z_list         = [];
    weight_list    = [];
    cluster_list   = [];
    subject_list   = [];
    type_list      = [];
    original_type_list = [];

    subjectGlobalID = 0;

    for k = 1:nClusters

        % cluster size: 2, 3, or 4 with equal probability
        nk = randsample([2 3 4], 1);

        % cluster-level latent effect
        Ak = sqrt(rho1) * randn;

        for i = 1:nk

            subjectGlobalID = subjectGlobalID + 1;

            % subject-level latent effect
            Bik = sqrt(rho2) * randn;

            for r_internal = 1:R

                % --------------------------------------------------------
                % internal type 1 = original type 1
                % internal type 2 = original type 3
                % --------------------------------------------------------
                if r_internal == 1
                    originalType = 1;
                else
                    originalType = 3;
                end

                % type-specific latent effect
                Cikr = sqrt(rho3) * randn;

                % independent event-type covariates
                Z1 = double(rand < 0.5);   % Bernoulli(0.5)
                Z2 = rand;                 % Uniform(0,1)
                Z  = [Z1, Z2];

                eta = Z * beta0;
                eta = max(min(eta, 50), -50);

                % censoring time
                Censor = min(nu1 + (nu2 - nu1) * rand, tau);

                complete_gaps = [];
                cumTime = 0;

                % Generate recurrent gaps until exceeding censoring time.
                % Use a safe upper bound to avoid accidental infinite loops.
                maxGapNumber = 10000;

                for jj = 1:maxGapNumber

                    % episode-level latent effect
                    Dijkr = sqrt(rho4) * randn;

                    W = Ak + Bik + Cikr + Dijkr;

                    % baseline T0 = -log(1 - Phi(W))
                    PhiW = normcdf(W);
                    PhiW = min(max(PhiW, 1e-12), 1 - 1e-12);
                    T0 = -log(1 - PhiW);

                    % ----------------------------------------------------
                    % Convert baseline gap T0 to observed gap T
                    % ----------------------------------------------------
                    if originalType == 1
                        % T0 = exp(eta) * T^2 / 2
                        gap = sqrt(2 * T0 * exp(-eta));

                    elseif originalType == 3
                        % T0 = exp(-eta) * log(1 + exp(eta)*T)
                        % exp(eta)*T0 = log(1 + exp(eta)*T)
                        % T = exp(-eta) * [exp(exp(eta)*T0) - 1]
                        arg = exp(eta) * T0;

                        % Prevent numerical overflow.
                        % If arg is huge, the gap must exceed censoring anyway.
                        if arg > 50
                            gap = inf;
                        else
                            gap = exp(-eta) * expm1(arg);
                        end

                    else
                        error('Unknown originalType.');
                    end

                    if cumTime + gap <= Censor
                        complete_gaps(end+1,1) = gap; %#ok<AGROW>
                        cumTime = cumTime + gap;
                    else
                        censored_gap = Censor - cumTime;
                        break;
                    end
                end

                if jj == maxGapNumber
                    warning('Maximum gap number reached. Check the data-generating setting.');
                    censored_gap = max(Censor - cumTime, 0);
                end

                % --------------------------------------------------------
                % Observed data construction:
                % If at least one complete gap is observed, keep complete
                % gaps only. The final censored gap is removed.
                % If no complete gap is observed, keep one censored gap.
                % --------------------------------------------------------
                nComplete = numel(complete_gaps);

                if nComplete > 0
                    S = nComplete;
                    row_times  = complete_gaps(:);
                    row_delta  = ones(S,1);
                else
                    S = 1;
                    row_times  = censored_gap;
                    row_delta  = 0;
                end

                row_weight = 1 / (nk * S);

                m = numel(row_times);

                time_list    = [time_list; row_times(:)];
                delta_list   = [delta_list; row_delta(:)];
                Z_list       = [Z_list; repmat(Z, m, 1)];
                weight_list  = [weight_list; repmat(row_weight, m, 1)];
                cluster_list = [cluster_list; repmat(k, m, 1)];
                subject_list = [subject_list; repmat(subjectGlobalID, m, 1)];
                type_list    = [type_list; repmat(r_internal, m, 1)];
                original_type_list = [original_type_list; repmat(originalType, m, 1)];
            end
        end
    end

    % ------------------------------------------------------------
    % Build data structure
    % ------------------------------------------------------------
    data.time      = time_list(:);
    data.delta     = delta_list(:);
    data.Z         = Z_list;
    data.weight    = weight_list(:);
    data.clusterID = cluster_list(:);
    data.subjectID = subject_list(:);
    data.typeID    = type_list(:);

    % Optional: record the original type label.
    % typeID=1 means original type1; typeID=2 means original type3.
    data.originalTypeID = original_type_list(:);

    data.R = R;
    data.p = p;
    data.nClusters = nClusters;
    data.typeSet = [1 3];
end