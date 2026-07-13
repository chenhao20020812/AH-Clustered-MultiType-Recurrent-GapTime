clear; clc;

load('PaperAligned_MIMIC_type123_baseline_CI_results.mat');

figure('Color','w','Position',[200 150 850 600]);
hold on;

colors = [
    0.0000 0.4470 0.7410
    0.8500 0.3250 0.0980
    0.4660 0.6740 0.1880
];

% 与传入代码保持一致的横轴上限
tau_plot = results.tau_plot;

for r = 1:3
    t = results.baseCI(r).t_grid(:);
    L = results.baseCI(r).Lambda_hat(:);

    if isempty(t)
        continue;
    end

    % 只保留 tau_plot 范围内的点
    keep = t <= tau_plot;
    t = t(keep);
    L = L(keep);

    % 让阶梯曲线延伸到横轴上限
    if t(end) < tau_plot
        t_plot = [0; t; tau_plot];
        L_plot = [0; L; L(end)];
    else
        t_plot = [0; t];
        L_plot = [0; L];
    end

    stairs(t_plot, L_plot, ...
        'LineWidth', 2.5, ...
        'Color', colors(r,:));
end

xlabel('Time (days)','FontSize',14);
ylabel('Baseline cumulative hazard','FontSize',14);

title('Baseline cumulative hazard functions','FontSize',15);

legend(results.typeNames, ...
       'Location','northwest', ...
       'FontSize',12, ...
       'Interpreter','none');

% 横轴数值及上限与传入代码对齐
xlim([0 tau_plot]);

grid on;
box on;

set(gca,'FontSize',13,'LineWidth',1.2);

hold off;

print(gcf, ...
    'PaperAligned_MIMIC_Lambda0_threeTypes_noCI', ...
    '-dpng','-r600');