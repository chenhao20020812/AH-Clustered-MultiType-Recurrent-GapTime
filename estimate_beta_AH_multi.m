function beta_hat = estimate_beta_AH_multi(data, tau, beta_init)
% estimate_beta_AH_multi
%
% Parameter-specific bounded optimization:
%
% beta1 in [-1.2,1.2]
% beta2 in [-0.8,0.8]
%
% beta_j = bound_j * tanh(theta_j)

    beta_init = beta_init(:);
    p = length(beta_init);

    % =========================================================
    % Parameter-specific bounds
    % =========================================================
    bound = [1.2;0.8];

    % near boundary threshold
    nearTol = 0.90 * bound;

    relaxFactor = 1.05;


    options = optimset('Display','off', ...
                       'MaxFunEvals',5e4, ...
                       'MaxIter',5e4, ...
                       'TolX',1e-7, ...
                       'TolFun',1e-8);


    % =========================================================
    % Starting values
    % =========================================================
    if p == 2

        starts_beta = [
            beta_init(:)';
            1.0  0.5;
            0.8  0.4;
            1.1  0.6;
            0.9  0.3;
            0.5  0.5;
            1.0  0.0;
            0.0  0.5;
            zeros(1,p)
        ];

    else

        starts_beta = [
            beta_init(:)';
            zeros(1,p);
            0.5*ones(1,p);
            ones(1,p)
        ];

    end


    % keep inside bounds
    starts_beta = max(min(starts_beta,...
                    (0.90*bound)'),...
                    -(0.90*bound)');


    nStart = size(starts_beta,1);


    cand_beta = zeros(p,nStart);
    cand_val  = inf(1,nStart);



    % =========================================================
    % Multi-start optimization
    % =========================================================

    for s = 1:nStart


        beta0_s = starts_beta(s,:)';


        theta0_s = beta_to_theta(beta0_s,bound);


        obj_theta = @(theta_row) ...
            objective_beta( ...
            theta_to_beta(theta_row(:),bound),...
            data,tau);


        theta_s = fminsearch(obj_theta,...
                             theta0_s(:)',...
                             options);


        beta_s = theta_to_beta(theta_s(:),bound);


        val_s = objective_beta(beta_s,data,tau);


        cand_beta(:,s)=beta_s;
        cand_val(s)=val_s;


    end



    % =========================================================
    % Select solution
    % =========================================================

    [best_val,best_idx]=min(cand_val);

    beta_best=cand_beta(:,best_idx);


    % non-boundary candidates

    is_boundary = abs(cand_beta) > nearTol;


    idx_nonbd = find(~any(is_boundary,1));


    if ~isempty(idx_nonbd)


        [best_nonbd_val,idx]=min(cand_val(idx_nonbd));


        beta_nonbd = cand_beta(:,idx_nonbd(idx));


        if best_nonbd_val <= relaxFactor*best_val

            beta_hat = beta_nonbd;

        else

            beta_hat = beta_best;

        end


    else

        beta_hat = beta_best;

    end



    % final restriction

    beta_hat = max(min(beta_hat,bound),-bound);


end



%% =========================================================
% transform functions
%% =========================================================

function beta = theta_to_beta(theta,bound)

theta = theta(:);

beta = bound .* tanh(theta);


end



function theta = beta_to_theta(beta,bound)

beta = beta(:);

x = beta ./ bound;

x=max(min(x,0.999999),-0.999999);

theta=atanh(x);


end



%% =========================================================
% objective
%% =========================================================

function val = objective_beta(beta,data,tau)


beta=beta(:);


if any(~isfinite(beta))

    val=1e12;

    return

end


U=score_AH_multi(beta,data,tau);


if any(~isfinite(U))

    val=1e12;

else

    val=norm(U)^2;

end


end