function data = simulate_dataset_AH_multi(nClusters, R, ...
    rho1, rho2, rho3, beta0, tau, nu1, nu2)
%--------------------------------------------------------------------------
% Simulate clustered multi-type recurrent gap-time data under the
% accelerated hazards model with type-specific baseline hazards.
%
% Model:
%   lambda_ijkr(t | Z_ikr) = lambda_0r( t * exp(beta0' * Z_ikr) )
%
% Type-specific baselines:
%   type 1: lambda_01(t) = t
%   type 2: lambda_02(t) = t^2
%
% Then:
%   type 1: T0_ijk1 = exp(beta0'Z_ik1) * T_ijk1^2 / 2
%   type 2: T0_ijk2 = exp(2*beta0'Z_ik2) * T_ijk2^3 / 3
%
% Latent effects:
%   A_k      ~ N(0, rho1)
%   B_ik     ~ N(0, rho2)
%   C_ikr    ~ N(0, rho3)
%   D_ijkr   ~ N(0, 1-rho1-rho2-rho3)
%
% Covariates:
%   Z_ikr = (Z1, Z2)'
%   Z1 ~ Bernoulli(0.5), Z2 ~ Uniform(0,1)
%
% Censoring:
%   C_ikr = min(U_ikr, tau), U_ikr ~ Uniform(nu1, nu2)
%
% Output follows the paper's observed data structure:
%   {X_ijkr, Delta_ikr, Z_ikr, j = 1,...,S_ikr}
%--------------------------------------------------------------------------

clusterSizeValues = [2 3 4];
clusterSizeProb   = [1/3 1/3 1/3];

rho4 = 1 - rho1 - rho2 - rho3;
if rho4 < 0
    error('Need rho1 + rho2 + rho3 <= 1.');
end

if R ~= 2
    warning(['Current simulation baseline is only specified for R = 2.\n' ...
             'For r > 2 you need to define lambda_0r(t) explicitly.']);
end

beta0 = beta0(:);

time_list      = {};
delta_list     = {};
Z_list         = {};
weight_list    = {};
clusterID_list = {};
subjectID_list = {};
typeID_list    = {};
gapID_list     = {};

subj_global_id = 0;

for k = 1:nClusters
    nk = randsample(clusterSizeValues, 1, true, clusterSizeProb);

    % cluster-level effect
    Ak = sqrt(rho1) * randn;

    for i = 1:nk
        subj_global_id = subj_global_id + 1;

        % subject-level effect within cluster
        Bik = sqrt(rho2) * randn;

        for r = 1:R
            % type-specific effect within subject
            Cikr = sqrt(rho3) * randn;

            % Z_ikr = (Z1, Z2)'
            Z1 = binornd(1, 0.5);
            Z2 = rand;
            Zikr = [Z1; Z2];

            eta = beta0' * Zikr;

            % censoring for this subject-type
            Uikr = nu1 + (nu2 - nu1) * rand;
            Cobs = min(Uikr, tau);

            t_current = 0;
            gap_times = [];
            gap_delta = [];

            j = 0;
            while true
                j = j + 1;

                % episode-specific effect
                Dijkr = sqrt(rho4) * randn;

                % latent baseline gap time:
                % T0_ijkr ~ Exp(1) via probability integral transform
                Xlatent = Ak + Bik + Cikr + Dijkr;
                PhiX    = normcdf(Xlatent);
                T0      = -log(1 - PhiX);

                % type-specific inversion:
                % T0_ijkr = Lambda_ijkr(T_ijkr)
                if r == 1
                    % type 1: lambda_01(t) = t
                    % lambda_ijk1(t) = lambda_01(t * exp(eta)) = t * exp(eta)
                    % Lambda_ijk1(t) = exp(eta) * t^2 / 2
                    % so Tgap = sqrt(2*T0 / exp(eta))
                    Tgap = sqrt(2 * T0 / exp(eta));

                elseif r == 2
                    % type 2: lambda_02(t) = t^2
                    % lambda_ijk2(t) = lambda_02(t * exp(eta)) = (t*exp(eta))^2
                    % Lambda_ijk2(t) = exp(2*eta) * t^3 / 3
                    % so Tgap = (3*T0 / exp(2*eta))^(1/3)
                    Tgap = (3 * T0 / exp(2 * eta))^(1/3);

                else
                    error('Baseline hazard is not specified for type r = %d.', r);
                end

                % accumulate recurrent gap times until censoring
                if t_current + Tgap <= Cobs
                    t_current = t_current + Tgap;
                    gap_times = [gap_times; Tgap];
                    gap_delta = [gap_delta; 1];
                else
                    % censored residual gap
                    T_cens = Cobs - t_current;
                    if T_cens < 0
                        T_cens = 0;
                    end
                    gap_times = [gap_times; T_cens];
                    gap_delta = [gap_delta; 0];
                    break;
                end
            end

            % Wang & Chang / Huang & Chen rule:
            % if M_ikr > 1, remove final censored gap and retain first M_ikr-1 complete gaps
            % if M_ikr = 1, retain the single censored gap
            Mikr = length(gap_times);

            if Mikr > 1
                Sikr = Mikr - 1;
                gap_times = gap_times(1:Sikr);
                gap_delta = gap_delta(1:Sikr);   % should all be 1
            else
                Sikr = 1;
            end

            % safety guard
            if isempty(gap_times)
                gap_times = 1e-8;
                gap_delta = 0;
                Sikr = 1;
            end

            % weight = 1 / (n_k * S_ikr)
            w = 1 / (nk * Sikr);
            nObs = length(gap_times);

            time_list      {end+1,1} = gap_times(:);
            delta_list     {end+1,1} = gap_delta(:);
            Z_list         {end+1,1} = repmat(Zikr(:)', nObs, 1);
            weight_list    {end+1,1} = w * ones(nObs,1);
            clusterID_list {end+1,1} = k * ones(nObs,1);
            subjectID_list {end+1,1} = subj_global_id * ones(nObs,1);
            typeID_list    {end+1,1} = r * ones(nObs,1);
            gapID_list     {end+1,1} = (1:nObs)';
        end
    end
end

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
end