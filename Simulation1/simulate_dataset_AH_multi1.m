function data = simulate_dataset_AH_multi1(nClusters, R, ...
    rho1, rho2, rho3, beta0, tau, nu1, nu2)
%--------------------------------------------------------------------------
% Simulate clustered multiple-type recurrent gap-time data
% under the accelerated hazards model:
%
%   lambda_ijkr(t | Z_ikr) = lambda_0r(t * exp(beta0' * Z_ikr))
%
% Event-type-specific baseline hazards:
%
%   Type 1:
%       lambda_01(t) = t
%       T0 = exp(eta) * T^2 / 2
%
%   Type 2:
%       lambda_02(t) = t^2
%       T0 = exp(2*eta) * T^3 / 3
%
%   Type 3:
%       lambda_03(t) = 1 / (1+t)
%       T0 = exp(-eta) * log(1 + exp(eta)*T)
%
% where eta = beta0' * Z_ikr.
%
% Latent effect:
%   X_ijkr = A_k + B_ik + C_ikr + D_ijkr
%
%   A_k    ~ N(0, rho1)
%   B_ik   ~ N(0, rho2)
%   C_ikr  ~ N(0, rho3)
%   D_ijkr ~ N(0, 1-rho1-rho2-rho3)
%
% Baseline gap:
%   T0_ijkr = -log(1 - Phi(X_ijkr))
%
% Covariates:
%   Z1 ~ Bernoulli(0.5)
%   Z2 ~ Uniform(0,1)
%
% Censoring:
%   C_ikr = min(U_ikr, tau), U_ikr ~ Uniform(nu1, nu2)
%
% Observed rule:
%   If M_ikr > 1, remove the last censored gap and keep the first M_ikr-1
%   complete gaps.
%
%   If M_ikr = 1, keep the single censored gap.
%--------------------------------------------------------------------------

% ============================================================
% Basic checks
% ============================================================

if R ~= 3
    warning('This simulation function is written for Table 2 with R = 3.');
end

rho4 = 1 - rho1 - rho2 - rho3;

if rho4 < 0
    error('Need rho1 + rho2 + rho3 <= 1.');
end

beta0 = beta0(:);

% ============================================================
% Cluster size distribution
% ============================================================

clusterSizeValues = [2 3 4];
clusterSizeProb   = [1/3 1/3 1/3];

% ============================================================
% Storage
% ============================================================

time_list      = {};
delta_list     = {};
Z_list         = {};
weight_list    = {};
clusterID_list = {};
subjectID_list = {};
typeID_list    = {};
gapID_list     = {};

subj_global_id = 0;

% ============================================================
% Generate data cluster by cluster
% ============================================================

for k = 1:nClusters

    % Cluster size
    nk = randsample(clusterSizeValues, 1, true, clusterSizeProb);

    % Cluster-level latent effect
    Ak = sqrt(rho1) * randn;

    for i = 1:nk

        subj_global_id = subj_global_id + 1;

        % Subject-level latent effect
        Bik = sqrt(rho2) * randn;

        for r = 1:R

            % Type-specific latent effect
            Cikr = sqrt(rho3) * randn;

            % ------------------------------------------------
            % Covariates Z_ikr = (Z1, Z2)'
            % ------------------------------------------------
            Z1 = binornd(1, 0.5);
            Z2 = rand;
            Zikr = [Z1; Z2];

            eta = beta0' * Zikr;

            % ------------------------------------------------
            % Censoring time
            % ------------------------------------------------
            Uikr = nu1 + (nu2 - nu1) * rand;
            Cobs = min(Uikr, tau);

            % ------------------------------------------------
            % Generate recurrent gap times until censoring
            % ------------------------------------------------
            t_current = 0;
            gap_times = [];
            gap_delta = [];

            while true

                % Episode-specific latent effect
                Dijkr = sqrt(rho4) * randn;

                % Latent standard normal mixture
                Xlatent = Ak + Bik + Cikr + Dijkr;

                % Probability integral transform
                PhiX = normcdf(Xlatent);

                % Numerical guard
                PhiX = min(max(PhiX, 1e-12), 1 - 1e-12);

                % Baseline gap time T0 ~ Exp(1)
                T0 = -log(1 - PhiX);

                % ------------------------------------------------
                % Type-specific inverse transformation
                % ------------------------------------------------
                if r == 1

                    % lambda_01(t) = t
                    % T0 = exp(eta) * T^2 / 2
                    Tgap = sqrt(2 * T0 / exp(eta));

                elseif r == 2

                    % lambda_02(t) = t^2
                    % T0 = exp(2*eta) * T^3 / 3
                    Tgap = (3 * T0 / exp(2 * eta))^(1/3);

                elseif r == 3

                    % lambda_03(t) = 1 / (1+t)
                    % T0 = exp(-eta) * log(1 + exp(eta)*T)
                    % Therefore:
                    % T = exp(-eta) * ( exp(exp(eta)*T0) - 1 )

                    expo_arg = exp(eta) * T0;

                    % Numerical guard to avoid overflow
                    expo_arg = min(expo_arg, 700);

                    Tgap = exp(-eta) * (exp(expo_arg) - 1);

                else
                    error('Baseline hazard is not specified for type r = %d.', r);
                end

                % ------------------------------------------------
                % Observe complete gap or censored residual gap
                % ------------------------------------------------
                if t_current + Tgap <= Cobs

                    t_current = t_current + Tgap;
                    gap_times = [gap_times; Tgap];
                    gap_delta = [gap_delta; 1];

                else

                    T_cens = Cobs - t_current;

                    if T_cens < 0
                        T_cens = 0;
                    end

                    gap_times = [gap_times; T_cens];
                    gap_delta = [gap_delta; 0];

                    break;
                end
            end

            % ------------------------------------------------
            % Observed data rule:
            %
            % If M_ikr > 1:
            %   remove final censored gap and keep first M_ikr - 1 gaps.
            %
            % If M_ikr = 1:
            %   retain the single censored gap.
            % ------------------------------------------------
            Mikr = length(gap_times);

            if Mikr > 1

                Sikr = Mikr - 1;

                gap_times = gap_times(1:Sikr);
                gap_delta = gap_delta(1:Sikr);

            else

                Sikr = 1;

            end

            % Safety guard
            if isempty(gap_times)
                gap_times = 1e-8;
                gap_delta = 0;
                Sikr = 1;
            end

            % ------------------------------------------------
            % Weight:
            %   1 / (n_k * S_ikr)
            % ------------------------------------------------
            w = 1 / (nk * Sikr);
            nObs = length(gap_times);

            time_list{end+1,1}      = gap_times(:);
            delta_list{end+1,1}     = gap_delta(:);
            Z_list{end+1,1}         = repmat(Zikr(:)', nObs, 1);
            weight_list{end+1,1}    = w * ones(nObs,1);
            clusterID_list{end+1,1} = k * ones(nObs,1);
            subjectID_list{end+1,1} = subj_global_id * ones(nObs,1);
            typeID_list{end+1,1}    = r * ones(nObs,1);
            gapID_list{end+1,1}     = (1:nObs)';
        end
    end
end

% ============================================================
% Convert to data structure
% ============================================================

data.time      = cell2mat(time_list);
data.delta     = cell2mat(delta_list);
data.Z         = cell2mat(Z_list);
data.weight    = cell2mat(weight_list);
data.clusterID = cell2mat(clusterID_list);
data.subjectID = cell2mat(subjectID_list);
data.typeID    = cell2mat(typeID_list);
data.gapID     = cell2mat(gapID_list);

data.R         = R;
data.beta0     = beta0;
data.tau       = tau;
data.nu1       = nu1;
data.nu2       = nu2;
data.rho       = [rho1 rho2 rho3 rho4];

end