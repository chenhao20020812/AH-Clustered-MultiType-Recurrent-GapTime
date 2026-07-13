function se = var_AH_resample_multi(beta_hat,data,tau,B)

beta_hat=beta_hat(:);

p=length(beta_hat);


clusters=unique(data.clusterID(:));

K=length(clusters);


beta_star=zeros(p,B);



% =========================================================
% Parameter-specific bounds
% =========================================================

bound=[1.2;0.8];

nearTol=0.90*bound;


options=optimset('Display','off',...
                 'MaxFunEvals',3e4,...
                 'MaxIter',3e4,...
                 'TolX',1e-7,...
                 'TolFun',1e-8);



% cluster influence

Dk_hat=compute_Dk_AH_multi(beta_hat,data,tau);


U_hat=score_AH_multi(beta_hat,data,tau);



for b=1:B


    G=randn(K,1);


    S=Dk_hat*G;



    obj_theta=@(theta_row) ...
        bootstrap_objective(...
        theta_to_beta(theta_row(:),bound),...
        data,tau,U_hat,S);



    theta0=beta_to_theta(beta_hat,bound);



    theta_b=fminsearch(obj_theta,...
                       theta0(:)',...
                       options);



    beta_star(:,b)=theta_to_beta(theta_b(:),bound);


end



se=sqrt(var(beta_star,0,2));


end




%% =========================================================
% bootstrap objective
%% =========================================================

function val=bootstrap_objective(beta,data,tau,U_hat,S)


beta=beta(:);


if any(~isfinite(beta))

    val=1e12;

    return

end


U=score_AH_multi(beta,data,tau);


if any(~isfinite(U))

    val=1e12;

else

    val=norm(U-U_hat-S)^2;

end


end




%% =========================================================
% transform
%% =========================================================

function beta=theta_to_beta(theta,bound)

theta=theta(:);

beta=bound.*tanh(theta);


end



function theta=beta_to_theta(beta,bound)


beta=beta(:);


x=beta./bound;


x=max(min(x,0.999999),-0.999999);


theta=atanh(x);


end
