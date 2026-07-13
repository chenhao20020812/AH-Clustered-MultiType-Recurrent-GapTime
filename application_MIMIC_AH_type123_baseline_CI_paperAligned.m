function results = application_MIMIC_AH_type123_baseline_CI_paperAligned(csvFile, B, tau_est, useParallel, initMatFile)
% application_MIMIC_AH_type123_baseline_CI_paperAligned
%
% Paper-aligned application code for:
%   Accelerated hazards model for clustered multiple-type recurrent gap times
%
% Paper application setting:
%   Data: MIMIC-IV recurrent hospitalizations
%   Type 1 = ICD-I, circulatory-system diseases
%   Type 2 = ICD-J, respiratory-system diseases
%   Type 3 = ICD-A, infectious/parasitic diseases
%
% Model:
%   lambda_ijkr(t | Z_ikr) = lambda_0r( t * exp(beta' Z_ikr) )
%
% Type-specific baseline cumulative hazards:
%   Lambda_01(t), Lambda_02(t), Lambda_03(t)
%
% Main outputs:
%   1. beta_hat, beta_star, SE, CI
%   2. Lambda_01(t), Lambda_02(t), Lambda_03(t)
%   3. 95% confidence bands for each type-specific baseline cumulative hazard
%
% Important paper-alignment step:
%   Your current CSV appears to have:
%       original type 1 count = 12949
%       original type 2 count = 5063
%       original type 3 count = 3870
%   But the paper writes:
%       paper type 1 = ICD-I count = 5063
%       paper type 2 = ICD-J count = 12949
%       paper type 3 = ICD-A count = 3870
%   Therefore, by default this code swaps original type 1 and original type 2:
%       original 1 -> paper 2
%       original 2 -> paper 1
%       original 3 -> paper 3
%
% Suggested call:
%   results = application_MIMIC_AH_type123_baseline_CI_paperAligned;
%
% Or:
%   results = application_MIMIC_AH_type123_baseline_CI_paperAligned( ...
%       'mimic_iv_AH_creatinine50_Z_age_gender_lactate_BUN.csv', ...
%       300, 365.25*4, true, 'MIMIC_Z12_type123_results.mat');

    if nargin < 1 || isempty(csvFile)
        csvFile = 'mimic_iv_AH_creatinine50_Z_age_gender_lactate_BUN.csv';
    end

    if nargin < 2 || isempty(B)
        B = 150;
    end

    if nargin < 3 || isempty(tau_est)
        tau_est = 365.25 * 4;       % 4 years if gap is measured in days
    end

    if nargin < 4 || isempty(useParallel)
        useParallel = true;
    end

    if nargin < 5 || isempty(initMatFile)
        initMatFile = 'MIMIC_Z12_type123_results.mat';
    end

    clc;
    fprintf('=============================================================\n');
    fprintf('Paper-aligned MIMIC-IV multi-type AH application\n');
    fprintf('Model: lambda_ijkr(t|Z) = lambda_0r(t exp(beta''Z))\n');
    fprintf('Paper type 1 = ICD-I / circulatory\n');
    fprintf('Paper type 2 = ICD-J / respiratory\n');
    fprintf('Paper type 3 = ICD-A / infectious\n');
    fprintf('CSV file: %s\n', csvFile);
    fprintf('Initial beta MAT file: %s\n', initMatFile);
    fprintf('Multiplier resampling B = %d\n', B);
    fprintf('Estimation tau = %.6f days = %.4f years\n', tau_est, tau_est/365.25);
    fprintf('useParallel = %d\n', useParallel);
    fprintf('=============================================================\n\n');

    rng(20260428, 'twister');

    %% ============================================================
    % 1. Build paper-aligned data
    % =============================================================
    data = build_multitype_data_from_MIMIC_paperAligned(csvFile);

    fprintf('Paper-aligned data summary:\n');
    fprintf('  Subjects       = %d\n', numel(unique(data.subjectID)));
    fprintf('  Clusters       = %d\n', numel(unique(data.clusterID)));
    fprintf('  Observations   = %d\n', numel(data.time));
    fprintf('  Event types    = %d\n', data.R);
    fprintf('  Covariates     = %d\n', size(data.Z,2));
    fprintf('  Type 1 rows    = %d  [ICD-I / circulatory]\n', sum(data.typeID == 1));
    fprintf('  Type 2 rows    = %d  [ICD-J / respiratory]\n', sum(data.typeID == 2));
    fprintf('  Type 3 rows    = %d  [ICD-A / infectious]\n', sum(data.typeID == 3));
    fprintf('  min/max gap    = %.6f / %.6f\n', min(data.time), max(data.time));
    fprintf('  events within tau_est = %d\n', sum(data.delta == 1 & data.time <= tau_est));
    fprintf('\n');

    %% ============================================================
    % 2. Load beta initial value from previous result
    % =============================================================
    p = size(data.Z,2);
    beta_init = load_beta_init_from_mat(initMatFile, p);

    fprintf('Initial beta used for optimization:\n');
    disp(beta_init);

    %% ============================================================
    % 3. Estimate beta
    % =============================================================
    fprintf('Estimating beta by minimizing ||U(beta)||^2 ...\n');

    beta_hat = estimate_beta_AH_multi_full(data, tau_est, beta_init);
    beta_hat = beta_hat(:);

    fprintf('\nEstimated beta:\n');
    disp(beta_hat);

    U_hat = score_AH_multi_full(beta_hat, data, tau_est);
    fprintf('||U(beta_hat)|| = %.8e\n\n', norm(U_hat));

    %% ============================================================
    % 4. Multiplier resampling for beta_star
    % =============================================================
    fprintf('Generating beta_star by multiplier resampling...\n');
    tic;

    [beta_star, Gmat, Dk_hat] = multiplier_beta_star_AH_multi_full( ...
        beta_hat, data, tau_est, B, useParallel);

    elapsed = toc;
    fprintf('Multiplier resampling finished in %.2f seconds.\n', elapsed);
    fprintf('Successful beta_star replicates = %d / %d\n\n', size(beta_star,2), B);

    se_hat = sqrt(var(beta_star, 0, 2));
    zval   = beta_hat ./ se_hat;
    pval   = 2 * (1 - normcdf(abs(zval)));
    CI_low = beta_hat - 1.96 * se_hat;
    CI_up  = beta_hat + 1.96 * se_hat;

    varNames = {'Age','Gender','Lactate','BUN'};

    fprintf('Regression results:\n');
    fprintf('--------------------------------------------------------------------------------\n');
    fprintf('%12s %12s %12s %12s %14s %24s\n', ...
        'Covariate','Estimate','SE','z','p-value','95% CI');
    fprintf('--------------------------------------------------------------------------------\n');

    for j = 1:length(beta_hat)
        fprintf('%12s %12.6f %12.6f %12.4f %14.6g   (%10.6f,%10.6f)\n', ...
            varNames{j}, beta_hat(j), se_hat(j), zval(j), pval(j), ...
            CI_low(j), CI_up(j));
    end

    fprintf('--------------------------------------------------------------------------------\n\n');

    %% ============================================================
    % 5. Determine plotting tau
    % =============================================================
    tmp_eta = data.Z * beta_hat;
    tmp_eta = max(min(tmp_eta, 50), -50);
    tmp_Xtilde = data.time .* exp(tmp_eta);

    transformed_event_times = tmp_Xtilde(data.delta == 1);
    transformed_event_times = transformed_event_times(isfinite(transformed_event_times) & transformed_event_times > 0);

    if isempty(transformed_event_times)
        error('No valid transformed event times for tau_plot.');
    end

    tau_q95 = quantile(transformed_event_times, 0.95);

    % Paper-aligned display:
    %   estimate beta using tau_est = 4 years,
    %   display baseline curves up to the non-sparse 95% transformed event-time region,
    %   but never beyond the 4-year tau.
    tau_plot = min(tau_est, tau_q95);

    fprintf('Plotting tau selection:\n');
    fprintf('  tau_est  = %.6f days = %.4f years\n', tau_est, tau_est/365.25);
    fprintf('  tau_q95  = %.6f transformed-time units\n', tau_q95);
    fprintf('  tau_plot = %.6f transformed-time units\n\n', tau_plot);

    %% ============================================================
    % 6. Type-specific baseline cumulative hazards + confidence bands
    % =============================================================
    fprintf('Estimating type-specific baseline cumulative hazards and 95%% confidence bands...\n');
    tic;

    baseCI = estimate_baseline_CI_multitype_AH_full( ...
        beta_hat, beta_star, Gmat, data, tau_plot);

    fprintf('Baseline CI calculation finished in %.2f seconds.\n\n', toc);

    %% ============================================================
    % 7. Plot and save
    % =============================================================
    outDir = fileparts(csvFile);
    if isempty(outDir)
        outDir = pwd;
    end

    typeNames = {'ICD-I / circulatory', ...
                 'ICD-J / respiratory', ...
                 'ICD-A / infectious'};

    figureNames = {'Figure3_Lambda01_ICD_I', ...
                   'Figure4_Lambda02_ICD_J', ...
                   'Figure5_Lambda03_ICD_A'};

    for r = 1:data.R
        fig = figure('Color','w','Position',[160,120,820,580]);
        hold on;

        t  = baseCI(r).t_grid(:);
        L  = baseCI(r).Lambda_hat(:);
        lo = baseCI(r).lower(:);
        up = baseCI(r).upper(:);

        if isempty(t)
            warning('Paper type %d has no event times within tau_plot.', r);
            title(sprintf('Paper type %d: no event times within tau_plot', r));
            continue;
        end

        stairs([0; t], [0; L],  'b-',  'LineWidth', 1.8);
        stairs([0; t], [0; up], 'r--', 'LineWidth', 1.3);
        stairs([0; t], [0; lo], '--',  'Color', [0.85 0.45 0.05], 'LineWidth', 1.3);

        xlabel('Transformed time t', 'FontSize', 13);
        ylabel(sprintf('\\Lambda_{0%d}(t)', r), 'FontSize', 13);

        title(sprintf('95%% confidence bands of \\Lambda_{0%d}(t) for type %d (%s)', ...
            r, r, typeNames{r}), 'FontSize', 13, 'Interpreter','tex');

        legend({'Estimate','Upper 95% band','Lower 95% band'}, ...
            'Location','northwest');

        xlim([0 tau_plot]);

        ymax = max(up);
        if isfinite(ymax) && ymax > 0
            ylim([0, 1.05*ymax]);
        end

        grid on; box on;
        set(gca, 'FontSize', 12, 'LineWidth', 1.1);

        fname_png = fullfile(outDir, sprintf('%s.png', figureNames{r}));
        fname_fig = fullfile(outDir, sprintf('%s.fig', figureNames{r}));

        saveas(fig, fname_png);
        saveas(fig, fname_fig);

        fprintf('Saved: %s\n', fname_png);
        fprintf('Saved: %s\n', fname_fig);
    end

    % Combined three-panel figure
    figAll = figure('Color','w','Position',[80,80,1500,470]);

    for r = 1:data.R
        subplot(1, data.R, r);
        hold on;

        t  = baseCI(r).t_grid(:);
        L  = baseCI(r).Lambda_hat(:);
        lo = baseCI(r).lower(:);
        up = baseCI(r).upper(:);

        if isempty(t)
            title(sprintf('Type %d: no data', r));
            continue;
        end

        stairs([0; t], [0; L],  'b-',  'LineWidth', 1.7);
        stairs([0; t], [0; up], 'r--', 'LineWidth', 1.2);
        stairs([0; t], [0; lo], '--',  'Color', [0.85 0.45 0.05], 'LineWidth', 1.2);

        xlabel('Transformed time t', 'FontSize', 12);
        ylabel(sprintf('\\Lambda_{0%d}(t)', r), 'FontSize', 12);
        title(sprintf('Type %d: %s', r, typeNames{r}), 'FontSize', 12);

        xlim([0 tau_plot]);

        ymax = max(up);
        if isfinite(ymax) && ymax > 0
            ylim([0, 1.05*ymax]);
        end

        grid on; box on;
        set(gca, 'FontSize', 11, 'LineWidth', 1.0);
    end

    sgtitle('Paper-aligned type-specific baseline cumulative hazards with 95% confidence bands', ...
        'FontSize', 14);

    fname_all_png = fullfile(outDir, 'PaperAligned_MIMIC_type123_Lambda0_CI_combined.png');
    fname_all_fig = fullfile(outDir, 'PaperAligned_MIMIC_type123_Lambda0_CI_combined.fig');

    saveas(figAll, fname_all_png);
    saveas(figAll, fname_all_fig);

    fprintf('Saved: %s\n', fname_all_png);
    fprintf('Saved: %s\n', fname_all_fig);

    %% ============================================================
    % 8. Save results
    % =============================================================
    results = struct();
    results.data        = data;
    results.beta_init   = beta_init;
    results.beta_hat    = beta_hat;
    results.beta_star   = beta_star;
    results.Gmat        = Gmat;
    results.Dk_hat      = Dk_hat;
    results.se_hat      = se_hat;
    results.zval        = zval;
    results.pval        = pval;
    results.CI_low      = CI_low;
    results.CI_up       = CI_up;
    results.baseCI      = baseCI;
    results.csvFile     = csvFile;
    results.B           = B;
    results.tau_est     = tau_est;
    results.tau_q95     = tau_q95;
    results.tau_plot    = tau_plot;
    results.initMatFile = initMatFile;
    results.typeNames   = typeNames;

    matName = fullfile(outDir, 'PaperAligned_MIMIC_type123_baseline_CI_results.mat');
    save(matName, 'results', '-v7.3');

    fprintf('\nSaved MAT results: %s\n', matName);
end


%% ========================================================================
% Build MIMIC multi-type data, aligned with paper type definition
% ========================================================================
function data = build_multitype_data_from_MIMIC_paperAligned(csvFile)

    T = readtable(csvFile);

    needVars = {'id','cluster','gap','type','Z1_ikr','Z2_ikr','Z3_ikr','Z4_ikr'};
    for j = 1:numel(needVars)
        if ~ismember(needVars{j}, T.Properties.VariableNames)
            error('Missing variable "%s" in CSV.', needVars{j});
        end
    end

    id_str = string(T.id);
    ok_id = ~ismissing(id_str) & strlength(id_str) > 0;

    cluster_raw = toDoubleColumn(T.cluster);
    gap_raw     = toDoubleColumn(T.gap);
    type_raw    = toDoubleColumn(T.type);
    Z1_raw      = toDoubleColumn(T.Z1_ikr);
    Z2_raw      = toDoubleColumn(T.Z2_ikr);
    Z3_raw      = toDoubleColumn(T.Z3_ikr);
    Z4_raw      = toDoubleColumn(T.Z4_ikr);

    ok_num = isfinite(cluster_raw) & isfinite(gap_raw) & isfinite(type_raw) & ...
             isfinite(Z1_raw) & isfinite(Z2_raw) & isfinite(Z3_raw) & isfinite(Z4_raw);

    T = T(ok_id & ok_num, :);

    id_str      = string(T.id);
    cluster_raw = toDoubleColumn(T.cluster);
    gap_raw     = toDoubleColumn(T.gap);
    type_raw    = toDoubleColumn(T.type);
    Z1_raw      = toDoubleColumn(T.Z1_ikr);
    Z2_raw      = toDoubleColumn(T.Z2_ikr);
    Z3_raw      = toDoubleColumn(T.Z3_ikr);
    Z4_raw      = toDoubleColumn(T.Z4_ikr);

    keep = gap_raw > 0 & type_raw >= 1 & type_raw <= 3 & ...
           isfinite(gap_raw) & isfinite(type_raw);

    T = T(keep, :);

    if ismember('event_time', T.Properties.VariableNames)
        try
            T = sortrows(T, {'id','event_time'});
        catch
            T = sortrows(T, {'id'});
        end
    else
        T = sortrows(T, {'id'});
    end

    id_str      = string(T.id);
    cluster_raw = toDoubleColumn(T.cluster);
    gap_raw     = toDoubleColumn(T.gap);
    type_raw    = toDoubleColumn(T.type);
    Z1_raw      = toDoubleColumn(T.Z1_ikr);
    Z2_raw      = toDoubleColumn(T.Z2_ikr);
    Z3_raw      = toDoubleColumn(T.Z3_ikr);
    Z4_raw      = toDoubleColumn(T.Z4_ikr);

    original_counts = zeros(3,1);
    for r = 1:3
        original_counts(r) = sum(type_raw == r);
    end

    fprintf('Original CSV type counts:\n');
    fprintf('  original type 1 = %d\n', original_counts(1));
    fprintf('  original type 2 = %d\n', original_counts(2));
    fprintf('  original type 3 = %d\n', original_counts(3));

    % Paper alignment:
    %   CSV currently appears to have original type 1 = J and original type 2 = I.
    %   Paper requires type 1 = I and type 2 = J.
    %
    % Therefore:
    %   original 1 -> paper 2
    %   original 2 -> paper 1
    %   original 3 -> paper 3
    typeID = zeros(size(type_raw));

    typeID(type_raw == 1) = 2;
    typeID(type_raw == 2) = 1;
    typeID(type_raw == 3) = 3;

    paper_counts = zeros(3,1);
    for r = 1:3
        paper_counts(r) = sum(typeID == r);
    end

    fprintf('Paper-aligned type counts:\n');
    fprintf('  paper type 1 = %d  [ICD-I / circulatory]\n', paper_counts(1));
    fprintf('  paper type 2 = %d  [ICD-J / respiratory]\n', paper_counts(2));
    fprintf('  paper type 3 = %d  [ICD-A / infectious]\n', paper_counts(3));

    [~,~,subjectID] = unique(id_str, 'stable');
    [~,~,clusterID] = unique(cluster_raw, 'stable');

    time = gap_raw(:);
    delta = ones(size(time));

    Z = [Z1_raw(:), Z2_raw(:), Z3_raw(:), Z4_raw(:)];

    R = 3;
    nObs = numel(time);
    weight = zeros(nObs,1);

    clusters = unique(clusterID);

    for a = 1:numel(clusters)
        k = clusters(a);
        idx_k = (clusterID == k);

        subj_k = unique(subjectID(idx_k));
        nk = numel(subj_k);

        for b = 1:numel(subj_k)
            sid = subj_k(b);

            for r = 1:R
                idx_ikr = idx_k & (subjectID == sid) & (typeID == r);
                Sikr = sum(idx_ikr);

                if Sikr > 0
                    weight(idx_ikr) = 1 / (nk * Sikr);
                end
            end
        end
    end

    if any(weight <= 0)
        error('Some observations have non-positive weights. Please check id/cluster/type structure.');
    end

    data = struct();
    data.time      = time(:);
    data.delta     = delta(:);
    data.Z         = Z;
    data.weight    = weight(:);
    data.clusterID = clusterID(:);
    data.subjectID = subjectID(:);
    data.typeID    = typeID(:);
    data.type_raw  = type_raw(:);
    data.R         = R;
end


function x = toDoubleColumn(x)

    if isnumeric(x)
        x = double(x);
        return;
    end

    if islogical(x)
        x = double(x);
        return;
    end

    if iscell(x)
        try
            x = str2double(string(x));
        catch
            x = cellfun(@double, x);
        end
        x = double(x);
        return;
    end

    if iscategorical(x)
        x = str2double(string(x));
        return;
    end

    if isstring(x) || ischar(x)
        x = str2double(string(x));
        return;
    end

    try
        x = str2double(string(x));
    catch
        error('Cannot convert input column to double.');
    end
end


%% ========================================================================
% Load initial beta from MAT file
% ========================================================================
function beta_init = load_beta_init_from_mat(initMatFile, p)

    beta_init = zeros(p,1);

    if ~exist(initMatFile, 'file')
        warning('Initial MAT file not found: %s. Use zeros as beta_init.', initMatFile);
        return;
    end

    S = load(initMatFile);
    beta_found = [];

    candidateNames = {'beta_hat','betaHat','betahat','beta','Beta_hat','BetaHat'};

    for a = 1:numel(candidateNames)
        nm = candidateNames{a};
        if isfield(S, nm)
            val = S.(nm);
            val = try_extract_beta_vector(val, p);
            if ~isempty(val)
                beta_found = val;
                break;
            end
        end
    end

    if isempty(beta_found) && isfield(S, 'results')
        val = try_extract_beta_from_struct(S.results, p);
        if ~isempty(val)
            beta_found = val;
        end
    end

    if isempty(beta_found)
        fns = fieldnames(S);
        for i = 1:numel(fns)
            val = S.(fns{i});
            beta_found = try_extract_beta_vector(val, p);
            if ~isempty(beta_found)
                break;
            end

            if isstruct(val)
                beta_found = try_extract_beta_from_struct(val, p);
                if ~isempty(beta_found)
                    break;
                end
            end
        end
    end

    if isempty(beta_found)
        warning('No beta vector of length %d found in %s. Use zeros as beta_init.', p, initMatFile);
    else
        beta_init = beta_found(:);
        fprintf('Loaded beta_init from %s.\n', initMatFile);
    end
end


function beta = try_extract_beta_from_struct(S, p)

    beta = [];

    if ~isstruct(S)
        return;
    end

    candidateNames = {'beta_hat','betaHat','betahat','beta','Beta_hat','BetaHat'};

    for a = 1:numel(candidateNames)
        nm = candidateNames{a};
        if isfield(S, nm)
            beta = try_extract_beta_vector(S.(nm), p);
            if ~isempty(beta)
                return;
            end
        end
    end

    fns = fieldnames(S);
    for i = 1:numel(fns)
        val = S.(fns{i});
        beta = try_extract_beta_vector(val, p);
        if ~isempty(beta)
            return;
        end
    end
end


function beta = try_extract_beta_vector(x, p)

    beta = [];

    if isnumeric(x)
        x = double(x);
        if isvector(x) && numel(x) == p && all(isfinite(x(:)))
            beta = x(:);
            return;
        end
    end

    if istable(x)
        try
            A = table2array(x);
            if isnumeric(A)
                beta = try_extract_beta_vector(A, p);
                if ~isempty(beta)
                    return;
                end
            end
        catch
        end
    end
end


%% ========================================================================
% Beta estimation
% ========================================================================
function beta_hat = estimate_beta_AH_multi_full(data, tau, beta_init)

    if nargin < 3 || isempty(beta_init)
        beta_init = zeros(size(data.Z,2),1);
    end

    beta_init = beta_init(:);

    options = optimset('Display','iter', ...
                       'MaxFunEvals', 1e5, ...
                       'MaxIter', 1e5, ...
                       'TolX', 1e-8, ...
                       'TolFun', 1e-8);

    obj = @(brow) score_objective_AH_multi_full(brow(:), data, tau);

    beta_hat_row = fminsearch(obj, beta_init(:)', options);
    beta_hat = beta_hat_row(:);
end


function val = score_objective_AH_multi_full(beta, data, tau)

    U = score_AH_multi_full(beta(:), data, tau);
    val = norm(U)^2;

    if ~isfinite(val)
        val = 1e100;
    end
end


%% ========================================================================
% Score U(beta)
% ========================================================================
function [U, Uk_list] = score_AH_multi_full(beta, data, tau)

    beta = beta(:);

    time      = data.time(:);
    delta     = data.delta(:);
    Z         = data.Z;
    weight    = data.weight(:);
    clusterID = data.clusterID(:);
    typeID    = data.typeID(:);
    R         = data.R;

    p = size(Z,2);

    clusters = unique(clusterID);
    K = numel(clusters);

    cluster_index = zeros(numel(clusterID),1);
    for kk = 1:K
        cluster_index(clusterID == clusters(kk)) = kk;
    end

    eta = Z * beta;
    eta = max(min(eta, 50), -50);

    Xtilde  = time .* exp(eta);
    e_minus = exp(-eta);

    U = zeros(p,1);
    Uk_list = zeros(p,K);

    tol = 1e-10;

    for r = 1:R
        idx_r = (typeID == r);

        t_event = unique(Xtilde(idx_r & delta == 1));

        if ~isempty(tau)
            t_event = t_event(t_event <= tau);
        end

        for m = 1:numel(t_event)
            t = t_event(m);

            Y = idx_r & (Xtilde >= t);
            if ~any(Y)
                continue;
            end

            wYe = weight(Y) .* e_minus(Y);
            denom = sum(wYe);

            if denom <= 0 || ~isfinite(denom)
                continue;
            end

            Zbar = (Z(Y,:)' * wYe) / denom;

            idxEv = idx_r & delta == 1 & abs(Xtilde - t) <= tol * max(1,abs(t));
            evList = find(idxEv);

            for a = 1:numel(evList)
                ii = evList(a);

                contrib = weight(ii) * (Z(ii,:)' - Zbar);

                U = U + contrib;

                kk = cluster_index(ii);
                Uk_list(:,kk) = Uk_list(:,kk) + contrib;
            end
        end
    end
end


%% ========================================================================
% D_k(beta_hat) for multiplier beta_star
% ========================================================================
function Dk = compute_Dk_AH_multi_full(beta, data, tau)

    beta = beta(:);

    time      = data.time(:);
    delta     = data.delta(:);
    Z         = data.Z;
    weight    = data.weight(:);
    clusterID = data.clusterID(:);
    typeID    = data.typeID(:);
    R         = data.R;

    p = size(Z,2);
    nObs = numel(time);

    clusters = unique(clusterID);
    K = numel(clusters);

    cluster_index = zeros(nObs,1);
    for kk = 1:K
        cluster_index(clusterID == clusters(kk)) = kk;
    end

    Dk = zeros(p,K);

    base = baseline_AH_multi_fast_full(beta, data, tau);

    eta = Z * beta;
    eta = max(min(eta, 50), -50);

    Xtilde  = time .* exp(eta);
    e_minus = exp(-eta);

    tol = 1e-10;

    for r = 1:R
        idx_r = (typeID == r);

        t_grid  = base(r).t_grid(:);
        dLambda = base(r).dLambda(:);

        for m = 1:numel(t_grid)
            t = t_grid(m);
            dLam = dLambda(m);

            Y = idx_r & (Xtilde >= t);
            if ~any(Y)
                continue;
            end

            wYe = weight(Y) .* e_minus(Y);
            denom = sum(wYe);

            if denom <= 0 || ~isfinite(denom)
                continue;
            end

            Zbar = (Z(Y,:)' * wYe) / denom;

            idx_at = find(Y);

            idxEvLocal = (delta(idx_at) == 1) & ...
                         (abs(Xtilde(idx_at) - t) <= tol * max(1,abs(t)));

            dM = double(idxEvLocal) - e_minus(idx_at) * dLam;

            for a = 1:numel(idx_at)
                ii = idx_at(a);
                kk = cluster_index(ii);

                contrib = weight(ii) * (Z(ii,:)' - Zbar) * dM(a);
                Dk(:,kk) = Dk(:,kk) + contrib;
            end
        end
    end
end


%% ========================================================================
% Multiplier beta_star
% ========================================================================
function [beta_star, Gmat, Dk_hat] = multiplier_beta_star_AH_multi_full( ...
    beta_hat, data, tau, B, useParallel)

    beta_hat = beta_hat(:);
    p = numel(beta_hat);

    clusters = unique(data.clusterID(:));
    K = numel(clusters);

    Dk_hat = compute_Dk_AH_multi_full(beta_hat, data, tau);

    beta_star_all = nan(p,B);
    Gmat_all = nan(K,B);

    options = optimset('Display','off', ...
                       'MaxFunEvals', 5e4, ...
                       'MaxIter', 5e4, ...
                       'TolX', 1e-7, ...
                       'TolFun', 1e-7);

    seeds = 20260428 + (1:B);

    if useParallel
        pool = gcp('nocreate');
        if isempty(pool)
            try
                parpool('local');
            catch ME
                warning('Could not start parallel pool: %s. Switching to serial.', ME.message);
                useParallel = false;
            end
        end
    end

    if useParallel
        parfor b = 1:B
            [beta_b, G_b] = one_multiplier_beta_star_AH_multi_full( ...
                b, seeds(b), beta_hat, data, tau, Dk_hat, options);

            beta_star_all(:,b) = beta_b;
            Gmat_all(:,b) = G_b;
        end
    else
        for b = 1:B
            [beta_b, G_b] = one_multiplier_beta_star_AH_multi_full( ...
                b, seeds(b), beta_hat, data, tau, Dk_hat, options);

            beta_star_all(:,b) = beta_b;
            Gmat_all(:,b) = G_b;
        end
    end

    ok = all(isfinite(beta_star_all),1) & all(isfinite(Gmat_all),1);
    beta_star = beta_star_all(:,ok);
    Gmat = Gmat_all(:,ok);

    if isempty(beta_star)
        error('All multiplier beta_star iterations failed.');
    end

    if size(beta_star,2) < max(30, ceil(0.5*B))
        warning('Only %d/%d beta_star iterations succeeded.', size(beta_star,2), B);
    end
end


function [beta_b, G] = one_multiplier_beta_star_AH_multi_full( ...
    b, seed, beta_hat, data, tau, Dk_hat, options)

    beta_hat = beta_hat(:);
    p = numel(beta_hat);

    clusters = unique(data.clusterID(:));
    K = numel(clusters);

    rng(seed, 'twister');
    G = randn(K,1);

    S = Dk_hat * G;

    obj = @(brow) multiplier_objective_AH_multi_full(brow(:), data, tau, S);

    try
        beta_row = fminsearch(obj, beta_hat(:)', options);
        beta_b = beta_row(:);

        if any(~isfinite(beta_b)) || numel(beta_b) ~= p
            beta_b = nan(p,1);
        end
    catch ME
        warning('beta_star iteration %d failed: %s', b, ME.message);
        beta_b = nan(p,1);
    end
end


function val = multiplier_objective_AH_multi_full(beta, data, tau, S)

    U = score_AH_multi_full(beta(:), data, tau);
    val = norm(U - S)^2;

    if ~isfinite(val)
        val = 1e100;
    end
end


%% ========================================================================
% Fast baseline cumulative hazard estimator
% ========================================================================
function base = baseline_AH_multi_fast_full(beta, data, tau)

    beta = beta(:);

    R = data.R;

    base = struct();

    for r = 1:R
        [t_event, dLam, Lam] = baseline_one_type_eventgrid_fast_full(beta, data, r, tau);

        base(r).t_grid = t_event(:);
        base(r).dLambda = dLam(:);
        base(r).Lambda0 = Lam(:);
    end
end


function [t_event, dLambda, Lambda] = baseline_one_type_eventgrid_fast_full(beta, data, r, tau)

    beta = beta(:);

    time   = data.time(:);
    delta  = data.delta(:);
    Z      = data.Z;
    weight = data.weight(:);
    typeID = data.typeID(:);

    eta = Z * beta;
    eta = max(min(eta, 50), -50);

    Xtilde  = time .* exp(eta);
    e_minus = exp(-eta);

    idx = (typeID == r);

    x  = Xtilde(idx);
    d  = delta(idx);
    w  = weight(idx);
    em = e_minus(idx);

    t_event = unique(x(d == 1));

    if ~isempty(tau)
        t_event = t_event(t_event <= tau);
    end

    if isempty(t_event)
        dLambda = zeros(0,1);
        Lambda = zeros(0,1);
        return;
    end

    riskWeight = w .* em;

    [xSort, ord] = sort(x);
    riskSort = riskWeight(ord);

    totalRisk = sum(riskWeight);
    ptr = 1;
    riskBelow = 0;

    dLambda = zeros(numel(t_event),1);

    tol = 1e-10;

    for m = 1:numel(t_event)
        t = t_event(m);

        while ptr <= numel(xSort) && xSort(ptr) < t - tol * max(1,abs(t))
            riskBelow = riskBelow + riskSort(ptr);
            ptr = ptr + 1;
        end

        denom = totalRisk - riskBelow;

        if denom <= 0 || ~isfinite(denom)
            dLambda(m) = 0;
            continue;
        end

        idxEv = (d == 1) & abs(x - t) <= tol * max(1,abs(t));
        num = sum(w(idxEv));

        dLambda(m) = num / denom;
    end

    Lambda = cumsum(dLambda);
end


function Lam_on_grid = baseline_one_type_on_fixed_grid_fast_full(beta, data, r, t_grid, tau)

    [t_event, ~, Lam_event] = baseline_one_type_eventgrid_fast_full(beta, data, r, tau);

    t_grid = t_grid(:);
    Lam_on_grid = zeros(numel(t_grid),1);

    if isempty(t_event)
        return;
    end

    ptr = 0;

    for m = 1:numel(t_grid)
        t = t_grid(m);

        while ptr < numel(t_event) && t_event(ptr+1) <= t
            ptr = ptr + 1;
        end

        if ptr > 0
            Lam_on_grid(m) = Lam_event(ptr);
        else
            Lam_on_grid(m) = 0;
        end
    end
end


%% ========================================================================
% Baseline confidence bands for each type
% ========================================================================
function baseCI = estimate_baseline_CI_multitype_AH_full( ...
    beta_hat, beta_star, Gmat, data, tau_plot)

    beta_hat = beta_hat(:);

    R = data.R;
    B = size(beta_star,2);

    base_hat = baseline_AH_multi_fast_full(beta_hat, data, tau_plot);

    baseCI = struct();

    for r = 1:R
        fprintf('  Type %d baseline CI...\n', r);

        t_grid = base_hat(r).t_grid(:);
        Lambda_hat = base_hat(r).Lambda0(:);
        dLambda_hat = base_hat(r).dLambda(:);

        M = numel(t_grid);

        if M == 0
            baseCI(r).t_grid = [];
            baseCI(r).Lambda_hat = [];
            baseCI(r).lower = [];
            baseCI(r).upper = [];
            baseCI(r).se = [];
            baseCI(r).delta_star = [];
            continue;
        end

        delta_star = nan(M,B);

        for b = 1:B
            beta_b = beta_star(:,b);
            G_b = Gmat(:,b);

            Lambda_beta_star = baseline_one_type_on_fixed_grid_fast_full( ...
                beta_b, data, r, t_grid, tau_plot);

            resid_process = residual_multiplier_baseline_one_type_full( ...
                beta_hat, data, r, t_grid, dLambda_hat, G_b);

            delta_star(:,b) = (Lambda_hat - Lambda_beta_star) + resid_process;
        end

        ok = all(isfinite(delta_star),1);
        delta_star = delta_star(:,ok);

        se = sqrt(var(delta_star, 0, 2));

        lower = Lambda_hat - 1.96 * se;
        upper = Lambda_hat + 1.96 * se;

        lower = max(lower, 0);

        lower_q = Lambda_hat - quantile(delta_star', 0.975)';
        upper_q = Lambda_hat - quantile(delta_star', 0.025)';
        lower_q = max(lower_q, 0);

        baseCI(r).t_grid = t_grid;
        baseCI(r).dLambda_hat = dLambda_hat;
        baseCI(r).Lambda_hat = Lambda_hat;
        baseCI(r).se = se;
        baseCI(r).lower = lower;
        baseCI(r).upper = upper;
        baseCI(r).lower_quantile = lower_q;
        baseCI(r).upper_quantile = upper_q;
        baseCI(r).delta_star = delta_star;
        baseCI(r).nBootUsed = size(delta_star,2);

        fprintf('    grid points = %d, bootstrap used = %d\n', M, size(delta_star,2));
    end
end


function resid_process = residual_multiplier_baseline_one_type_full( ...
    beta_hat, data, r, t_grid, dLambda_hat, G)

    beta_hat = beta_hat(:);
    G = G(:);

    time      = data.time(:);
    delta     = data.delta(:);
    Z         = data.Z;
    weight    = data.weight(:);
    clusterID = data.clusterID(:);
    typeID    = data.typeID(:);

    clusters = unique(clusterID);
    K = numel(clusters);

    if numel(G) ~= K
        error('Length of G must equal number of clusters.');
    end

    cluster_index = zeros(numel(clusterID),1);
    for kk = 1:K
        cluster_index(clusterID == clusters(kk)) = kk;
    end

    eta = Z * beta_hat;
    eta = max(min(eta, 50), -50);

    Xtilde  = time .* exp(eta);
    e_minus = exp(-eta);

    idx = (typeID == r);

    x  = Xtilde(idx);
    d  = delta(idx);
    w  = weight(idx);
    em = e_minus(idx);

    cidx_all = cluster_index(idx);

    t_grid = t_grid(:);
    dLambda_hat = dLambda_hat(:);

    M = numel(t_grid);
    resid_process = zeros(M,1);

    if M == 0
        return;
    end

    riskWeight = w .* em;
    riskWeightG = riskWeight .* G(cidx_all);

    [xSort, ord] = sort(x);
    riskSort = riskWeight(ord);
    riskGSort = riskWeightG(ord);

    totalRisk = sum(riskWeight);
    totalRiskG = sum(riskWeightG);

    ptr = 1;
    riskBelow = 0;
    riskGBelow = 0;

    cum = 0;
    tol = 1e-10;

    for m = 1:M
        t = t_grid(m);

        while ptr <= numel(xSort) && xSort(ptr) < t - tol * max(1,abs(t))
            riskBelow = riskBelow + riskSort(ptr);
            riskGBelow = riskGBelow + riskGSort(ptr);
            ptr = ptr + 1;
        end

        denom = totalRisk - riskBelow;
        riskG = totalRiskG - riskGBelow;

        if denom <= 0 || ~isfinite(denom)
            resid_process(m) = cum;
            continue;
        end

        idxEv = (d == 1) & abs(x - t) <= tol * max(1,abs(t));

        eventG = sum(w(idxEv) .* G(cidx_all(idxEv)));

        dLam = dLambda_hat(m);

        inc = (eventG - dLam * riskG) / denom;

        cum = cum + inc;
        resid_process(m) = cum;
    end
end