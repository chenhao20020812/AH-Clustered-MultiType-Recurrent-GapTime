function out = modelcheck_MIMIC_AH_betaStar_parallel(matFile, B)
% modelcheck_MIMIC_AH_betaStar_parallel
% Full model checking with beta_star, parallel version.
%
% Test statistic:
%   U = sup_{t,z} |F(t,z; beta_hat)|
%
% Bootstrap process:
%   Fhat_b(t,z)
%     = {F(t,z; beta_hat) - F(t,z; beta_star_b)}
%       + n^{-1/2} sum_k G_k * centered_residual_k(t,z; beta_hat)
%
% Default:
%   B = 300

    if nargin < 1 || isempty(matFile)
        matFile = 'MIMIC_Z12_type123_results.mat';
    end
    if nargin < 2 || isempty(B)
        B = 300;
    end

    clc;
    fprintf('=============================================\n');
    fprintf('Model checking for MIMIC-IV AH model\n');
    fprintf('Full beta_star version with parallel bootstrap\n');
    fprintf('B = %d\n', B);
    fprintf('=============================================\n');

    S = load(matFile);
    results = S.results;

    beta_hat = results.beta_hat(:);
    data     = results.data;
    tau      = [];

    fprintf('\nLoaded beta_hat:\n');
    disp(beta_hat);

    fprintf('Subjects  = %d\n', numel(unique(data.subjectID)));
    fprintf('Clusters  = %d\n', numel(unique(data.clusterID)));
    fprintf('Obs       = %d\n', numel(data.time));
    fprintf('Types     = %d\n', data.R);

    %% start parallel pool
    pool = gcp('nocreate');

    if isempty(pool)
        pool = parpool('local', 32);
    end
    
    % 防止长时间运行中途自动关闭
    try
        pool.IdleTimeout = Inf;
    catch
        warning('Could not set IdleTimeout to Inf.');
    end

    %% 1. checking grid
    fprintf('\nBuilding checking grid...\n');
    [t_grid, Zcut] = make_checking_grid(beta_hat, data);

    fprintf('t-grid points = %d\n', length(t_grid));
    fprintf('z-grid points = %d\n', size(Zcut,1));

    %% 2. observed residual process
    fprintf('\nComputing observed residual process F(t,z; beta_hat)...\n');
    F_obs = compute_F_process_AH_multi(beta_hat, data, tau, t_grid, Zcut);
    U_obs = max(abs(F_obs(:)));

    fprintf('Observed U = %.6f\n', U_obs);

    %% 3. D_k(beta_hat)
    fprintf('\nComputing D_k(beta_hat)...\n');
    Dk_hat = compute_Dk_AH_multi(beta_hat, data, tau);

    clusters = unique(data.clusterID(:));
    K = length(clusters);
    p = length(beta_hat);

    beta_star = nan(p, B);
    U_boot    = nan(B, 1);

    options = optimset('Display','off', ...
                       'MaxFunEvals', 5e4, ...
                       'MaxIter', 5e4);

    baseSeed = 20260422;
    seeds = baseSeed + (1:B);

    fprintf('\nStarting parallel bootstrap with beta_star...\n');
    tic;

    parfor b = 1:B
        try
            rng(seeds(b), 'twister');

            G = randn(K,1);
            Sg = Dk_hat * G;

            obj = @(beta_row) norm(score_AH_multi(beta_row(:), data, tau) - Sg)^2;

            beta_b_row = fminsearch(obj, beta_hat(:)', options);
            beta_b = beta_b_row(:);

            beta_star(:,b) = beta_b;

            F_star = compute_F_process_AH_multi(beta_b, data, tau, t_grid, Zcut);

            F_mult = compute_centered_multiplier_process_AH_multi( ...
                beta_hat, data, tau, t_grid, Zcut, G);

            F_boot = (F_obs - F_star) + F_mult;

            U_boot(b) = max(abs(F_boot(:)));

        catch ME
            warning('Bootstrap %d failed: %s', b, ME.message);
            beta_star(:,b) = nan(p,1);
            U_boot(b) = nan;
        end
    end

    runTime = toc;

    ok = isfinite(U_boot);
    U_boot_ok = U_boot(ok);

    if isempty(U_boot_ok)
        error('All bootstrap iterations failed.');
    end

    if sum(ok) < B
        warning('Only %d/%d bootstrap iterations succeeded.', sum(ok), B);
    end

    p_value = mean(U_boot_ok >= U_obs);

    fprintf('\n=============================================\n');
    fprintf('Goodness-of-fit test finished\n');
    fprintf('Successful bootstrap = %d / %d\n', sum(ok), B);
    fprintf('U_obs   = %.6f\n', U_obs);
    fprintf('p-value = %.6f\n', p_value);
    fprintf('Time    = %.2f minutes\n', runTime / 60);
    fprintf('=============================================\n');

    out = struct();
    out.U_obs       = U_obs;
    out.p_value     = p_value;
    out.U_boot      = U_boot;
    out.U_boot_ok   = U_boot_ok;
    out.beta_star   = beta_star;
    out.beta_hat    = beta_hat;
    out.t_grid      = t_grid;
    out.Zcut        = Zcut;
    out.success     = ok;
    out.runTime     = runTime;
    out.B           = B;

    save('MIMIC_modelcheck_betaStar_parallel_B300_results.mat', 'out');

    fprintf('\nSaved: MIMIC_modelcheck_betaStar_parallel_B300_results.mat\n');
end


function [t_grid, Zcut] = make_checking_grid(beta, data)

    beta = beta(:);

    time  = data.time(:);
    delta = data.delta(:);
    Z     = data.Z;

    eta = Z * beta;
    eta = max(min(eta, 50), -50);

    Xtilde = time .* exp(eta);

    % ============================================================
    % 1. t-grid: transformed event times, downsample to maxT = 50
    % ============================================================
    t_grid = unique(Xtilde(delta == 1));
    t_grid = t_grid(isfinite(t_grid) & t_grid > 0);

    maxT = 50;

    if length(t_grid) > maxT
        idx = round(linspace(1, length(t_grid), maxT));
        idx = unique(idx);
        t_grid = t_grid(idx);

        % 如果 unique 后不足 50 个点，不强行补；保持稳定
        if length(t_grid) > maxT
            t_grid = t_grid(1:maxT);
        end
    end

    % ============================================================
    % 2. z-grid: covariate cut points
    % ============================================================
    % 先构造相对充分的候选网格，然后统一抽样到 maxZ = 30。
    %
    % 对连续变量用 25%, 50%, 75% 分位数；
    % 对二分类变量，如 gender，只保留其唯一值。
    %
    % 由于 Z 有 4 个协变量：
    %   age, gender, lactate, BUN
    % 原来大概是 2 * 3 * 3 * 3 = 54 个 z-grid；
    % 现在从候选 z-grid 中均匀抽取最多 30 个。
    % ============================================================

    q_list = [0.25 0.50 0.75];

    p = size(Z,2);
    z_grid = cell(p,1);

    for j = 1:p
        zj = Z(:,j);
        zj = zj(isfinite(zj));

        if isempty(zj)
            error('Covariate column %d has no finite values.', j);
        end

        u = unique(zj);

        % 如果是二分类或离散取值很少，直接用唯一值
        if numel(u) <= 3
            z_grid{j} = u(:)';
        else
            qj = quantile(zj, q_list);
            qj = unique(qj);
            z_grid{j} = qj(:)';
        end
    end

    Zcut_all = make_cartesian_grid(z_grid);

    % 删除重复行，防止 gender 或分位数重复导致冗余
    Zcut_all = unique(Zcut_all, 'rows');

    maxZ = 30;

    if size(Zcut_all,1) > maxZ
        idx = round(linspace(1, size(Zcut_all,1), maxZ));
        idx = unique(idx);

        Zcut = Zcut_all(idx,:);

        % 如果 unique 后超过 maxZ，再截断
        if size(Zcut,1) > maxZ
            Zcut = Zcut(1:maxZ,:);
        end
    else
        Zcut = Zcut_all;
    end

    fprintf('Reduced checking grid generated:\n');
    fprintf('  maxT target = %d, actual t-grid = %d\n', maxT, length(t_grid));
    fprintf('  maxZ target = %d, actual z-grid = %d\n', maxZ, size(Zcut,1));
end


function F = compute_F_process_AH_multi(beta, data, tau, t_grid, Zcut)

    beta = beta(:);

    time      = data.time(:);
    delta     = data.delta(:);
    Z         = data.Z;
    weight    = data.weight(:);
    clusterID = data.clusterID(:);
    typeID    = data.typeID(:);
    R         = data.R;

    clusters = unique(clusterID);
    K = length(clusters);

    eta     = Z * beta;
    Xtilde  = time .* exp(eta);
    e_minus = exp(-eta);

    Mt = length(t_grid);
    Mz = size(Zcut,1);

    Fk = zeros(K, Mt, Mz);

    base = baseline_AH_multi(beta, data, tau);

    tol = 1e-10;

    for r = 1:R
        idx_r = (typeID == r);

        tb_grid = base(r).t_grid(:);
        dLam    = base(r).dLambda(:);

        for m = 1:length(tb_grid)
            tb = tb_grid(m);
            dLambda_m = dLam(m);

            t_pos = find(t_grid >= tb);
            if isempty(t_pos)
                continue;
            end

            Y = idx_r & (Xtilde >= tb);
            idxY = find(Y);

            if isempty(idxY)
                continue;
            end

            dN_all = double(idx_r & delta == 1 & abs(Xtilde - tb) < tol);
            dM = dN_all(idxY) - e_minus(idxY) * dLambda_m;

            for zz = 1:Mz
                zcut = Zcut(zz,:);
                Iz = all(Z(idxY,:) <= zcut, 2);

                if ~any(Iz)
                    continue;
                end

                idxYZ = idxY(Iz);
                dM_z  = dM(Iz);
                w_z   = weight(idxYZ);

                for a = 1:length(idxYZ)
                    obs = idxYZ(a);
                    kk = find(clusters == clusterID(obs), 1);

                    contrib = w_z(a) * dM_z(a);

                    Fk(kk, t_pos, zz) = Fk(kk, t_pos, zz) + contrib;
                end
            end
        end
    end

    F = squeeze(sum(Fk, 1)) / sqrt(K);
end


function F_mult = compute_centered_multiplier_process_AH_multi( ...
    beta, data, tau, t_grid, Zcut, G)

    beta = beta(:);

    time      = data.time(:);
    delta     = data.delta(:);
    Z         = data.Z;
    weight    = data.weight(:);
    clusterID = data.clusterID(:);
    typeID    = data.typeID(:);
    R         = data.R;

    clusters = unique(clusterID);
    K = length(clusters);

    eta     = Z * beta;
    Xtilde  = time .* exp(eta);
    e_minus = exp(-eta);

    Mt = length(t_grid);
    Mz = size(Zcut,1);

    Fk = zeros(K, Mt, Mz);

    base = baseline_AH_multi(beta, data, tau);

    tol = 1e-10;

    for r = 1:R
        idx_r = (typeID == r);

        tb_grid = base(r).t_grid(:);
        dLam    = base(r).dLambda(:);

        for m = 1:length(tb_grid)
            tb = tb_grid(m);
            dLambda_m = dLam(m);

            t_pos = find(t_grid >= tb);
            if isempty(t_pos)
                continue;
            end

            Y = idx_r & (Xtilde >= tb);
            idxY = find(Y);

            if isempty(idxY)
                continue;
            end

            denom = sum(weight(idxY) .* e_minus(idxY));
            if denom <= 0
                continue;
            end

            dN_all = double(idx_r & delta == 1 & abs(Xtilde - tb) < tol);
            dM = dN_all(idxY) - e_minus(idxY) * dLambda_m;

            for zz = 1:Mz
                zcut = Zcut(zz,:);

                Iz = all(Z(idxY,:) <= zcut, 2);

                num_z = sum(weight(idxY) .* e_minus(idxY) .* double(Iz));
                ratio = num_z / denom;

                centered = double(Iz) - ratio;
                contrib_vec = weight(idxY) .* centered .* dM;

                for a = 1:length(idxY)
                    obs = idxY(a);
                    kk = find(clusters == clusterID(obs), 1);

                    Fk(kk, t_pos, zz) = Fk(kk, t_pos, zz) + contrib_vec(a);
                end
            end
        end
    end

    F_mult = zeros(Mt, Mz);

    for kk = 1:K
        F_mult = F_mult + G(kk) * squeeze(Fk(kk,:,:));
    end

    F_mult = F_mult / sqrt(K);
end


function Zcut = make_cartesian_grid(z_grid)

    p = length(z_grid);
    grids = cell(1,p);

    [grids{:}] = ndgrid(z_grid{:});

    n = numel(grids{1});
    Zcut = zeros(n,p);

    for j = 1:p
        Zcut(:,j) = grids{j}(:);
    end
end
