%% table_compare_multitype_Jiang.m
% Multi-type comparison:
% Proposed clustered AH vs extended Jiang-type working-independence method
% Based on current paper:
%   type 1: lambda01(t)=t
%   type 2: lambda02(t)=t^2
%   T0_ijkr = -log(1-Phi(Ak + Bik + Cikr + Dijkr))
%   comparison covariates:
%       Z_k1 ~ Unif(0,1), shared within cluster for type 1
%       Z_k2 ~ Bernoulli(0.5), shared within cluster for type 2
%
% Scalar beta0 is used here to match the comparison formulas in the paper.

clear; clc;
rng(2025);

tau   = 3;
beta0 = 1;

rho_triplets = [
    0.1 0.1 0.1;
    0.3 0.3 0.3
];

nu1 = 0.5; 
nu2 = 2.5;

nClusters_set = [50 100];
nRep  = 1000;
Bboot = 500;

if isempty(gcp('nocreate')), parpool; end

Results = struct();

for rr = 1:size(rho_triplets,1)
    rho1 = rho_triplets(rr,1);
    rho2 = rho_triplets(rr,2);
    rho3 = rho_triplets(rr,3);

    fprintf('\n====================================================\n');
    fprintf('Multi-type compare | beta0=%.2f | rho=(%.2f, %.2f, %.2f)\n', ...
        beta0, rho1, rho2, rho3);
    fprintf('====================================================\n');

    for ii = 1:length(nClusters_set)
        nClusters = nClusters_set(ii);
        fprintf('>>> nClusters=%d, nRep=%d, B=%d\n', nClusters, nRep, Bboot);

        betaC = zeros(nRep,1); seC = zeros(nRep,1); cpC = zeros(nRep,1);
        betaJ = zeros(nRep,1); seJ = zeros(nRep,1); cpJ = zeros(nRep,1);

        parfor rep = 1:nRep
            % ------------------------------------------------------------
            % Simulate under comparison setup in the current multi-type paper
            % ------------------------------------------------------------
            data = simulate_dataset_multitype_compare( ...
                nClusters, rho1, rho2, rho3, beta0, tau, nu1, nu2);

            % ---------------- Proposed clustered method ------------------
            beta_hat_C = solve_beta_by_fminsearch( ...
                @(b) U_AH_multitype_1d(b, data, true), beta0);

            se_hat_C   = var_AH_multiplier_fminsearch_multitype_1d( ...
                beta_hat_C, data, Bboot, "cluster", true);

            betaC(rep) = beta_hat_C;
            seC(rep)   = se_hat_C;
            cpC(rep)   = (beta_hat_C - 1.96*se_hat_C <= beta0) && ...
                         (beta0 <= beta_hat_C + 1.96*se_hat_C);

            % ---------------- Extended Jiang-type method ----------------
            % Working independence:
            % - no cluster weight 1/nk
            % - resample by subject-type unit
            dataJ = build_data_jiang_multitype(data);

            beta_hat_J = solve_beta_by_fminsearch( ...
                @(b) U_AH_multitype_1d(b, dataJ, false), beta0);

            se_hat_J   = var_AH_multiplier_fminsearch_multitype_1d( ...
                beta_hat_J, dataJ, Bboot, "subjtype", false);

            betaJ(rep) = beta_hat_J;
            seJ(rep)   = se_hat_J;
            cpJ(rep)   = (beta_hat_J - 1.96*se_hat_J <= beta0) && ...
                         (beta0 <= beta_hat_J + 1.96*se_hat_J);
        end

        outC.bias  = mean(betaC - beta0);
        outC.seEmp = std(betaC);
        outC.ese   = mean(seC);
        outC.cp    = mean(cpC);

        outJ.bias  = mean(betaJ - beta0);
        outJ.seEmp = std(betaJ);
        outJ.ese   = mean(seJ);
        outJ.cp    = mean(cpJ);

        fprintf('\n----- n=%d | rho=(%.2f, %.2f, %.2f) -----\n', ...
            nClusters, rho1, rho2, rho3);
        fprintf('Proposed: Bias=% .4f | SE=% .4f | ESE=% .4f | CP=%.3f\n', ...
            outC.bias, outC.seEmp, outC.ese, outC.cp);
        fprintf('JiangExt: Bias=% .4f | SE=% .4f | ESE=% .4f | CP=%.3f\n', ...
            outJ.bias, outJ.seEmp, outJ.ese, outJ.cp);

        key = sprintf('MT_beta1_rho1_%02d_rho2_%02d_rho3_%02d_n_%d', ...
            round(100*rho1), round(100*rho2), round(100*rho3), nClusters);

        Results.(key).Proposed  = outC;
        Results.(key).JiangExt  = outJ;
        Results.(key).beta0     = beta0;
        Results.(key).tau       = tau;
        Results.(key).rho1      = rho1;
        Results.(key).rho2      = rho2;
        Results.(key).rho3      = rho3;
        Results.(key).nClusters = nClusters;
        Results.(key).nRep      = nRep;
        Results.(key).Bboot     = Bboot;
    end
end

save('Table_compare_multitype_Jiang.mat','Results');
fprintf('\nSaved: Table_compare_multitype_Jiang.mat\n');

%% ========================= functions =========================

function data = simulate_dataset_multitype_compare(nClusters, rho1, rho2, rho3, beta0, tau, nu1, nu2)
% Multi-type comparison DGP:
%   T0_ijkr = -log(1 - Phi(Ak + Bik + Cikr + Dijkr))
%   type 1: lambda01(t)=t      => T = sqrt(2*T0*exp(-eta))
%   type 2: lambda02(t)=t^2    => T = (3*T0*exp(-2*eta))^(1/3)
%
% comparison covariates:
%   Z_k1 ~ Unif(0,1)
%   Z_k2 ~ Bern(0.5)
%
% all subjects within the same cluster share the same covariate value
% for each event type.

clusterSizeValues = [2 3 4];
clusterSizeProb   = [1/3 1/3 1/3];

rho4 = 1 - rho1 - rho2 - rho3;
if rho4 < 0
    error('Need rho1 + rho2 + rho3 <= 1.');
end

time_list      = {};
delta_list     = {};
Z_list         = {};
clusterID_list = {};
subjectID_list = {};
typeID_list    = {};

for k = 1:nClusters
    nk = randsample(clusterSizeValues,1,true,clusterSizeProb);
    Ak = sqrt(rho1) * randn;

    % comparison covariates: cluster-level and type-specific
    Zk1 = rand;           % U(0,1)
    Zk2 = double(rand < 0.5);   % Bernoulli(0.5)

    for i = 1:nk
        Bik = sqrt(rho2) * randn;

        for r = 1:2
            Cikr = sqrt(rho3) * randn;
            Uikr = nu1 + (nu2 - nu1)*rand;
            Cens = min(Uikr, tau);

            if r == 1
                Zkr = Zk1;
            else
                Zkr = Zk2;
            end

            t_current = 0;
            gap_times = [];
            gap_delta = [];

            while true
                Dijkr = sqrt(rho4) * randn;
                X     = Ak + Bik + Cikr + Dijkr;
                PhiX  = 0.5*(1 + erf(X/sqrt(2)));
                PhiX  = min(max(PhiX, 1e-12), 1-1e-12);
                T0    = -log(1 - PhiX);

                eta = beta0 * Zkr;

                if r == 1
                    % lambda01(t)=t => Lambda01(t)=e^eta * t^2 / 2
                    Tgap = sqrt(2 * T0 * exp(-eta));
                else
                    % lambda02(t)=t^2 => Lambda02(t)=e^(2eta) * t^3 / 3
                    Tgap = (3 * T0 * exp(-2*eta))^(1/3);
                end

                if t_current + Tgap <= Cens
                    t_current = t_current + Tgap;
                    gap_times = [gap_times; Tgap];
                    gap_delta = [gap_delta; 1];
                else
                    T_cens = Cens - t_current;
                    if T_cens < 0, T_cens = 0; end
                    gap_times = [gap_times; T_cens];
                    gap_delta = [gap_delta; 0];
                    break;
                end
            end

            Mikr = length(gap_times);
            if Mikr > 1
                Sikr = Mikr - 1;
                gap_times = gap_times(1:Sikr);
                gap_delta = gap_delta(1:Sikr);
            else
                Sikr = 1;
            end

            nObs = length(gap_times);

            time_list      {end+1,1} = gap_times(:);
            delta_list     {end+1,1} = gap_delta(:);
            Z_list         {end+1,1} = Zkr * ones(nObs,1);
            clusterID_list {end+1,1} = k   * ones(nObs,1);
            subjectID_list {end+1,1} = i   * ones(nObs,1);
            typeID_list    {end+1,1} = r   * ones(nObs,1);
        end
    end
end

data.time      = cell2mat(time_list);
data.delta     = cell2mat(delta_list);
data.Z         = cell2mat(Z_list);
data.clusterID = cell2mat(clusterID_list);
data.subjectID = cell2mat(subjectID_list);
data.typeID    = cell2mat(typeID_list);
end

function dataJ = build_data_jiang_multitype(data)
% Extended Jiang working-independence version:
% - remove 1/nk cluster weight
% - treat each (subject, type) process as a separate working unit
cid = data.clusterID(:);
sid = data.subjectID(:);
rid = data.typeID(:);

[~,~,subjTypeID] = unique([cid sid rid],'rows');
Sikr = accumarray(subjTypeID, 1);

dataJ = data;
dataJ.clusterID = subjTypeID;   % working unit id
dataJ.subjectID = subjTypeID;
dataJ.typeID    = rid;
dataJ.weight    = 1 ./ Sikr(subjTypeID);
end

function beta_hat = solve_beta_by_fminsearch(Ufun, beta_init)
obj = @(b) (Ufun(b)).^2;
options = optimset('Display','off', 'MaxFunEvals',5e4, 'MaxIter',5e4);
beta_hat = fminsearch(obj, beta_init(:)', options);
beta_hat = beta_hat(:);
end

function U = U_AH_multitype_1d(beta, data, useClusterWeight)
% Multi-type estimating function:
% U(beta)=sum_r sum_{ijk} int (Z - Zbar_r)dN_tilde
beta = beta(:);

time      = data.time(:);
delta     = data.delta(:);
Z         = data.Z(:);
clusterID = data.clusterID(:);
subjectID = data.subjectID(:);
typeID    = data.typeID(:);

eta     = Z * beta;
Xtilde  = time .* exp(eta);
e_minus = exp(-eta);

[nk_vec, Sikr_vec] = get_nk_Sikr(clusterID, subjectID, typeID);

if useClusterWeight
    w0 = 1 ./ (nk_vec .* Sikr_vec);
else
    if isfield(data, 'weight')
        w0 = data.weight(:);
    else
        w0 = 1 ./ Sikr_vec;
    end
end

U = 0;
types = unique(typeID);

for rr = 1:length(types)
    r = types(rr);

    idxr = (typeID == r);
    evT  = unique(Xtilde(idxr & delta==1));

    for m = 1:length(evT)
        t = evT(m);

        Y = idxr & (Xtilde >= t);
        if ~any(Y), continue; end

        wY = w0 .* Y .* e_minus;
        denom = sum(wY);
        if denom <= 0, continue; end

        Zbar_r = sum(Z .* wY) / denom;

        idxEv = idxr & (Xtilde == t) & (delta == 1);
        U = U + sum(w0(idxEv) .* (Z(idxEv) - Zbar_r));
    end
end
end

function se = var_AH_multiplier_fminsearch_multitype_1d(beta_hat, data, B, unit, useClusterWeight)
beta_hat = beta_hat(:);
p = length(beta_hat);
if p ~= 1
    error('This function is for 1D beta only.');
end

switch lower(unit)
    case {'cluster','clusters'}
        units = unique(data.clusterID(:));
    case {'subjtype','subjecttype','subject-type'}
        cid = data.clusterID(:);
        sid = data.subjectID(:);
        rid = data.typeID(:);
        units = unique(1e12*cid + 1e6*sid + rid);
    otherwise
        error('unit must be "cluster" or "subjtype".');
end

K = length(units);
beta_star = zeros(p, B);

options = optimset('Display','off', 'MaxFunEvals',5e4, 'MaxIter',5e4);

D_hat = compute_D_unit_multitype_1d(beta_hat, data, unit, useClusterWeight);
scorefun = @(b) U_AH_multitype_1d(b, data, useClusterWeight);

for b = 1:B
    G = randn(K,1);
    S = D_hat * G;

    obj = @(beta_row) norm(scorefun(beta_row(:)) - S)^2;
    beta0_row  = beta_hat(:)';
    beta_b_row = fminsearch(obj, beta0_row, options);

    beta_star(:,b) = beta_b_row(:);
end

se = sqrt(var(beta_star, 0, 2));
end

function D = compute_D_unit_multitype_1d(beta, data, unit, useClusterWeight)
time      = data.time(:);
delta     = data.delta(:);
Z         = data.Z(:);
clusterID = data.clusterID(:);
subjectID = data.subjectID(:);
typeID    = data.typeID(:);

eta     = Z * beta;
Xtilde  = time .* exp(eta);
e_minus = exp(-eta);

[nk_vec, Sikr_vec] = get_nk_Sikr(clusterID, subjectID, typeID);

if useClusterWeight
    w0 = 1 ./ (nk_vec .* Sikr_vec);
else
    if isfield(data,'weight')
        w0 = data.weight(:);
    else
        w0 = 1 ./ Sikr_vec;
    end
end

switch lower(unit)
    case "cluster"
        units = unique(clusterID);
        uid = clusterID;
    case {"subjtype","subjecttype","subject-type"}
        uid = 1e12*clusterID + 1e6*subjectID + typeID;
        units = unique(uid);
    otherwise
        error('unit must be "cluster" or "subjtype".');
end

K = length(units);
D = zeros(1,K);

[t_grid_cell, dLam_cell] = baseline_jump_multitype(beta, data, w0);

types = unique(typeID);

for rr = 1:length(types)
    r = types(rr);

    t_grid = t_grid_cell{r};
    dLam   = dLam_cell{r};

    for m = 1:length(t_grid)
        t = t_grid(m);

        Y = (typeID==r) & (Xtilde >= t);
        if ~any(Y), continue; end

        wY = w0 .* Y .* e_minus;
        denom = sum(wY);
        if denom <= 0, continue; end

        Zbar_r = sum(Z .* wY) / denom;

        idx = find(Y);
        dN  = double((Xtilde(idx)==t) & (delta(idx)==1));
        dM  = dN - e_minus(idx) * dLam(m);

        for s = 1:length(idx)
            ii = idx(s);
            k = find(units == uid(ii), 1);
            if isempty(k), continue; end
            D(k) = D(k) + w0(ii) * (Z(ii) - Zbar_r) * dM(s);
        end
    end
end
end

function [t_grid_cell, dLam_cell] = baseline_jump_multitype(beta, data, w0)
time   = data.time(:);
delta  = data.delta(:);
Z      = data.Z(:);
typeID = data.typeID(:);

eta     = Z * beta;
Xtilde  = time .* exp(eta);
e_minus = exp(-eta);

R = max(typeID);
t_grid_cell = cell(R,1);
dLam_cell   = cell(R,1);

for r = 1:R
    idxr = (typeID == r);
    t_grid = unique(Xtilde(idxr & delta==1));
    dLam   = zeros(length(t_grid),1);

    for m = 1:length(t_grid)
        t = t_grid(m);
        Y = idxr & (Xtilde >= t);

        denom = sum(w0 .* Y .* e_minus);
        if denom <= 0
            dLam(m) = 0;
            continue;
        end

        num = sum(w0 .* idxr .* (Xtilde==t) .* (delta==1));
        dLam(m) = num / denom;
    end

    t_grid_cell{r} = t_grid;
    dLam_cell{r}   = dLam;
end
end

function [nk_vec, Sikr_vec] = get_nk_Sikr(clusterID, subjectID, typeID)
n = length(clusterID);
nk_vec   = zeros(n,1);
Sikr_vec = zeros(n,1);

clusters = unique(clusterID);

for kk = 1:length(clusters)
    k = clusters(kk);
    idxk = (clusterID == k);
    subs = unique(subjectID(idxk));
    nk   = length(subs);

    for ii = 1:length(subs)
        sid = subs(ii);

        for r = unique(typeID(idxk & subjectID==sid))'
            idx = idxk & (subjectID == sid) & (typeID == r);
            Sikr = sum(idx);

            nk_vec(idx)   = nk;
            Sikr_vec(idx) = Sikr;
        end
    end
end
end
