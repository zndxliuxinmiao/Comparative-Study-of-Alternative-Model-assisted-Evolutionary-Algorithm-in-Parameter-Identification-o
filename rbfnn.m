clc;
clear;
%load comsol P2D model
model = mphload('model.mph');

%load real data
load realdata001C;
load realdata05C;
load realdata1C;
load realdata2C;
load realdata3C_new;

%load training samples
load population;

%% physical 3C
Dimen = 6;
% population quantity
NP = 100;

e = 0.0001;
% LCB judgement's weight
ww = 1;

minfv_p2d = 10000;
G = 60;

% circulation condition
DE = 1;
% Initial mutation operator
F = 0.9;
% crossover operator 
CR = 0.5;
% maximum number of iterations 
DE_MAX = 40;
% Upper and lower boundary of parameters (subject achievements)
xmin = [7.5e-11 8.5e-14 3e-13 8e-12  8e-12  0.3];
xmax = [2.0e-10 2.0e-13 7e-13 4e-11  4e-11  0.5];

% 最优个体的集合
visited_pbest = [];
visited_value = [];

h=waitbar(0,'DE is starting');
pause(1);

tic
while DE
    % initial population
    x = zeros(NP,Dimen);
    % Variation population
    v = zeros(NP,Dimen);
    % Select population
    u = zeros(NP,Dimen);
    % evaluation function
    fv = zeros(NP,1);
    fvu = zeros(NP,1);
    minfv = 10000;
    % initialze  LHS
    X_temp = (xmax - xmin);
    X1 = lhsdesign(NP, Dimen);
    for i=1:NP
        for j=1:Dimen
            x(i, j) = X1(i, j) * X_temp(j) + xmin(j);
        end
    end
    
    % bulid RBFNN model
    net = newrb(A.', B.');
    for m=1:NP
        temp_objF = sim(net, x(m,:).');
        fv(m) = temp_objF;
    end
    
    % iterative caculations
    h2=waitbar(0,'start to DE');
    % mutation
    for gen = 1:G
        % Adaptive mutation operator
        for m = 1:NP
            r1 = randi([1,NP]);
            while(r1 == m)
                r1 = randi([1,NP]);
            end
            r2 = randi([1,NP]);
            while(r2 == m) || (r2 == r1)
                r2 = randi([1,NP]);
            end
            r3 = randi([1,NP]);
            while(r3 == m) || (r3 == r1) || (r3 == r2)
                r3 = randi([1,NP]);
            end
            v(m,:) = x(r1,:) + F*(x(r2,:)-x(r3,:));
        end
    
    
        % crossover
        for i = 1: NP
            r = randi([1,Dimen]);
            cr = rand(1);
            for n = 1: Dimen
                if (cr <= CR) || (n==r)   
                    u(i,n) = v(i,n);  
                else  
                    u(i,n) = x(i,n);  
                end  
            end  
        end
    
        % Processing of boundary conditions
        for i = 1: NP
            for j = 1:Dimen
                if u(i,j) > xmax(j)
                    u(i,j) = xmax(j);
                elseif u(i,j) < xmin(j)
                    u(i,j) = xmin(j);
                end
            end
        end
    
        % selection option
        for m=1:NP
            temp_objF = sim(net, x(m,:).');
            fvu(m) = temp_objF;
            if fvu(m) < fv(m)
                x(m, :) = u(m, :);
                fv(m) = fvu(m);
            end
            % select the best
            if fv(m) < minfv
                pbest = x(m,:);
                minfv = fv(m);
            end
        end
        waitbar(gen/G,h2,['teaching and learning have completed' num2str((gen/G)*100) '%']);
    end
    disp(pbest);
    disp(minfv);
    close(h2);
    %% combine P2D model
    try
        [temp_fv,~,rmse] =fitness_d(pbest,model);
    catch ErrorInfo
        continue
    end
    disp(rmse)
    visited_pbest = [visited_pbest; pbest];
    visited_value = [visited_value; temp_fv];    
    if ( rmse <= e )
        pbestd = pbest;
        break;
    end
    if (DE == DE_MAX )
        break;
    end
    if (rmse <= minfv_p2d)
        minfv_p2d = rmse;
        pbestd = pbest;
    end
    
    A = [A;pbest];
    B = [B;temp_fv];

    waitbar(DE/DE_MAX,h,['DE has completed ' num2str(DE) ' times ']);
    DE = DE+1;
end

toc

close(h);

[temp_fv,phis,rmse]=fitness_d(pbestd,model);

n = length(realdata3C);

xreal=10*length(realdata3C)-(10-1);
xsimu=10*length(phis)-(10-1);

plot(1:50:xreal,realdata3C(1:5:n),'ob','linewidth',2);
hold on
plot(1:10:xsimu,phis,'r','linewidth',2);

legend('realdata','simudata');
xlabel('time/s');
ylabel('voltage/V');
title([' The  RMSE is ',num2str(rmse)]);

figure_FontSize=10;
set(findobj('FontSize',10),'FontSize',figure_FontSize);
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
set(0,'defaultfigurecolor','w');
print(gcf,'-dbmp','I:\d.bmp')