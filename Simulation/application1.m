function results = application_MIMIC_AH_eventZ_Z12_type123(csvFile, B)
% application_MIMIC_AH_eventZ_Z12_type123
% MIMIC-IV AH model
% - types 1,2,3 only
% - event-level covariates Z1_event + Z2_event
%
% Required variables in csv:
%   id, cluster, gap, type, Z1_event, Z2_event

    if nargin < 1 || isempty(csvFile)
        csvFile = '/Users/chenhao/Documents/科研方向3/mimic_iv_AH_creatinine50_Z_age_gender_lactate_BUN.csv';
    end
    if nargin < 2 || isempty(B)
        B = 300;
    end

    clc;
    fprintf('=============================================\n');
    fprintf('MIMIC-IV AH model with event-level covariates\n');
    fprintf('Types used: 1,2,3 only\n');
    fprintf('Covariates used: age + gender + lactate + BUN\n');
    fprintf('CSV file: %s\n', csvFile);
    fprintf('Multiplier resampling B = %d\n', B);
    fprintf('=============================================\n');

    %% 1. build data
    data = build_multitype_data_from_MIMIC_eventZ_Z12_type123(csvFile);

    fprintf('\nData summary:\n');
    fprintf('  Subjects  = %d\n', numel(unique(data.subjectID)));
    fprintf('  Clusters  = %d\n', numel(unique(data.clusterID)));
    fprintf('  Obs       = %d\n', numel(data.time));
    fprintf('  Types     = %d\n', data.R);

    for r = 1:data.R
        fprintf('  Type %d count: %d\n', r, sum(data.typeID == r));
    end

    %% 2. estimate beta
    beta_init = zeros(size(data.Z,2),1);

    fprintf('\nEstimating beta...\n');
    beta_hat = estimate_beta_AH_multi(data, [], beta_init);
    beta_hat = beta_hat(:);

    fprintf('Estimated beta:\n');
    disp(beta_hat);

    %% 3. estimate SE by parallel multiplier resampling
    fprintf('\nEstimating standard errors by parallel multiplier resampling...\n');
    tic;
    se_hat = var_AH_resample_multi1(beta_hat, data, [], B);
    se_hat = se_hat(:);
    fprintf('Parallel resampling finished in %.2f seconds.\n', toc);

    zval    = beta_hat ./ se_hat;
    pval    = 2 * (1 - normcdf(abs(zval)));
    CI_low  = beta_hat - 1.96 * se_hat;
    CI_high = beta_hat + 1.96 * se_hat;

    varNames = {'age','gender','lactate','BUN'};

    fprintf('\nRegression results:\n');
    fprintf('-----------------------------------------------------------------\n');
    fprintf('%18s %12s %12s %12s %12s %18s\n', ...
        'Covariate','Estimate','SE','z','p-value','95% CI');
    fprintf('-----------------------------------------------------------------\n');
    for j = 1:length(beta_hat)
        fprintf('%18s %12.6f %12.6f %12.4f %12.4g [%8.6f,%8.6f]\n', ...
            varNames{j}, beta_hat(j), se_hat(j), zval(j), pval(j), ...
            CI_low(j), CI_high(j));
    end
    fprintf('-----------------------------------------------------------------\n');

    %% 4. estimate type-specific baseline cumulative hazards
    fprintf('\nEstimating type-specific baseline cumulative hazards...\n');
    base = baseline_AH_multi(beta_hat, data, []);

    outDir = fileparts(csvFile);

    for r = 1:data.R
        figure('Color','w','Position',[180,120,760,560]); hold on;

        t_grid = base(r).t_grid(:);
        Lam    = base(r).Lambda0(:);

        if isempty(t_grid)
            warning('Type %d has no event times.', r);
            title(sprintf('Type %d: no event times', r), 'FontSize', 13);
            hold off;
            continue;
        end

        stairs([0; t_grid], [0; Lam], 'LineWidth', 2.0);
        xlabel('Transformed time t', 'FontSize', 13);
        ylabel(sprintf('\\Lambda_{0,%d}(t)', r), 'FontSize', 13);
        title(sprintf('MIMIC-IV data: baseline cumulative hazard, type %d', r), ...
            'FontSize', 13, 'Interpreter', 'none');
        grid on; box on;
        set(gca, 'FontSize', 12, 'LineWidth', 1.2);

        fname = fullfile(outDir, sprintf('MIMIC_Z12_type123_baseline_type%d.png', r));
        saveas(gcf, fname);
        fprintf('Saved figure: %s\n', fname);

        hold off;
    end

    %% 5. save results
    results = struct();
    results.beta_hat = beta_hat;
    results.se_hat   = se_hat;
    results.zval     = zval;
    results.pval     = pval;
    results.CI_low   = CI_low;
    results.CI_high  = CI_high;
    results.base     = base;
    results.data     = data;

    save(fullfile(outDir, 'MIMIC_Z12_type123_results.mat'), 'results');
    fprintf('\nSaved results: %s\n', ...
        fullfile(outDir, 'MIMIC_Z12_type123_results.mat'));
end


function data = build_multitype_data_from_MIMIC_eventZ_Z12_type123(csvFile)
% build_multitype_data_from_MIMIC_eventZ_Z12_type123
% MIMIC-IV real data
% - only type 1,2,3
% - only event-level covariates Z1_event and Z2_event

    T = readtable(csvFile);

    needVars = {'id','cluster','gap','type','Z1_ikr','Z2_ikr','Z3_ikr','Z4_ikr'};
    for j = 1:numel(needVars)
        if ~ismember(needVars{j}, T.Properties.VariableNames)
            error('Missing variable "%s".', needVars{j});
        end
    end

    ok_id = ~ismissing(string(T.id));

    ok_num = ~isnan(double(T.cluster)) & ...
         ~isnan(double(T.gap)) & ...
         ~isnan(double(T.type)) & ...
         ~isnan(double(T.Z1_ikr)) & ...
         ~isnan(double(T.Z2_ikr)) & ...
         ~isnan(double(T.Z3_ikr)) & ...
         ~isnan(double(T.Z4_ikr));

    T = T(ok_id & ok_num, :);
    T = T(T.gap > 0, :);
    T = T(T.type >= 1 & T.type <= 3, :);

    if ismember('event_time', T.Properties.VariableNames)
        try
            T = sortrows(T, {'id','event_time'});
        catch
            T = sortrows(T, {'id'});
        end
    else
        T = sortrows(T, {'id'});
    end

    [~,~,subjectID] = unique(string(T.id), 'stable');
    [~,~,clusterID] = unique(T.cluster, 'stable');
    [~,~,typeID]    = unique(T.type, 'stable');

    time  = double(T.gap);
    delta = ones(size(time));

    Z = [
    double(T.Z1_ikr), ...
    double(T.Z2_ikr), ...
    double(T.Z3_ikr), ...
    double(T.Z4_ikr)
        ];
    R = max(typeID);

    nObs = height(T);
    weight = zeros(nObs,1);

    clusters = unique(clusterID);
    for kk = 1:numel(clusters)
        k = clusters(kk);
        idx_k = (clusterID == k);

        subj_k = unique(subjectID(idx_k));
        nk = numel(subj_k);

        for ii = 1:numel(subj_k)
            sid = subj_k(ii);

            for r = 1:R
                idx_ikr = idx_k & (subjectID == sid) & (typeID == r);
                Sikr = sum(idx_ikr);

                if Sikr > 0
                    weight(idx_ikr) = 1 / (nk * Sikr);
                end
            end
        end
    end

    data = struct();
    data.time      = time(:);
    data.delta     = delta(:);
    data.Z         = Z;
    data.weight    = weight(:);
    data.clusterID = clusterID(:);
    data.subjectID = subjectID(:);
    data.typeID    = typeID(:);
    data.R         = R;
end
