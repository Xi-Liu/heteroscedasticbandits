clear all;
clc;

% assign values to constants
d = 4;
C1 = 1;
C2 = 1;
C3 = 1;
C4 = 1;
L = 1.1;
lambda = 1;
delta = 0.1;
sigmaMax2 = 2*L;
Mf = 1;
alpha2 = sqrt(2*d*L*L*(1+log(C1/delta)/C2*log(C1/delta)/C2));
alpha3 = sqrt(2*d*L*log(d/delta));

% for arm context distribution
numActions = 10;
actionMus = {};
actionSigmas = {};
actionMus{1} = [1, 5];
actionMus{2} = [2, 6];
actionMus{3} = [3, 7];
actionMus{4} = [4, 8];
actionMus{5} = [5, 9];
actionMus{6} = [6, 10];
actionMus{7} = [7, 15];
actionMus{8} = [8, 20];
actionMus{9} = [2, 10];
actionMus{10} = [10, 20];

actionSigmas{1} = [10 5; 5 3];
actionSigmas{2} = [9 6; 6 5];
actionSigmas{3} = [6 7; 7 10];
actionSigmas{4} = [5 8; 8 16];
actionSigmas{5} = [15 9; 9 7];
actionSigmas{6} = [16 10; 10 15];
actionSigmas{7} = [10 11; 11 15];
actionSigmas{8} = [12 6; 6 5];
actionSigmas{9} = [7 4; 4 6];
actionSigmas{10} = [4 3; 3 4];

% for user context distribution
userMu = [12 23];
userSigma = [10 7; 7 5];

% for context distribution
sContexts = [];
sRewards = [];

% for parameters of reward distribution
theta = [0.6 0.3 0.2 0.65];
phi = [0.2 0.4 0.8 0.3];

% total number of users
T = 50000;
% total number of interactions used in calculation
K = 4.5;

% for reproducibility
rng default  
initialRounds = 5;
for i = 1:initialRounds
    beta = unifrnd(-3,5);
    contexts = {};
    % the user context
    x1 = mvnrnd(userMu,userSigma,1);
    % concatenate user context with action context
    for j = 1:numActions
        x2 = mvnrnd(actionMus{j},actionSigmas{j},1);
        x = [x1 x2];
        x = x/norm(x);
        contexts{j} = x;   
    end
    
    rewardMus ={};
    rewardSigmas = {};
    
    for j = 1:numActions
        rewardMus{j} = dot(contexts{j}, theta);
        rewardSigmas{j} = dot(contexts{j},phi)+L;
    end
    
    live = true;
    while (live == true) && (length(sRewards) < K*i)
        armChosen = randi(numActions);
        sContexts = [sContexts; contexts{armChosen}];
        observedReward = mvnrnd(rewardMus{armChosen},rewardSigmas{armChosen},1);
        sRewards = [sRewards; observedReward];
        if observedReward < beta
            live = false;
        else
            live = true;
        end
    end
end

% generate rewards on the fly
total_repeat = 10;

for repeat = 1 : total_repeat
    
    accumuRegrets = [];
    
    for i = 1:T
        % already perform some rounds before it really starts
        % t = initialRounds - 1 + i;
        
        t = length(sRewards);

        alpha1 = sigmaMax2 * sqrt(d * log((t+lambda)/(lambda * delta)))+sqrt(lambda);
        
        % gradually reduce the vlaue of delta
        deltaTemp = delta / (t^2);

        rho = Mf * alpha1 * deltaTemp / 3 * (1 + 2 * alpha3 * deltaTemp / 3) + sqrt(lambda) + alpha2 * deltaTemp / 3;

        xi = C3 * alpha1 + C4 * rho * deltaTemp;

        beta = unifrnd(-3,5);
        contexts = {};
        % the user context
        x1 = mvnrnd(userMu,userSigma,1);
        % concatenate user context with action context
        for j = 1:numActions
            x2 = mvnrnd(actionMus{j},actionSigmas{j},1);
            x = [x1 x2];
            x = x/norm(x);
            contexts{j} = x;
        end

        rewardMus ={};
        rewardSigmas = {};

        for j = 1:numActions
            rewardMus{j} = dot(contexts{j}, theta);
            rewardSigmas{j} = dot(contexts{j},phi)+L;
        end
        live = true;

        % the optimal policy
        armOptimal = optimalPolicy(beta, theta, phi, L, contexts);
        % the HR_UCB policy
        armChosen = HR_UCB(beta, xi, d, lambda, L, contexts, sContexts,sRewards);

        individualRegret = 0;
        
        accumObservedRewards = 0;
        while (live == true)
            observedReward = mvnrnd(rewardMus{armChosen},rewardSigmas{armChosen},1);
            if length(sRewards) < K*(i+initialRounds)
                sContexts = [sContexts; contexts{armChosen}];
                sRewards = [sRewards; observedReward];
            end
            accumObservedRewards = accumObservedRewards + observedReward;
            if observedReward < beta
                live = false;
            else
                live = true;
            end
        end
        
%         live = true;
%         accumOptimalRewards = 0;
%         while (live == true)
%             optimalReward = mvnrnd(rewardMus{armOptimal},rewardSigmas{armOptimal},1);
%             accumOptimalRewards = accumOptimalRewards + optimalReward;
%             if optimalReward < beta
%                 live = false;
%             else
%                 live = true;
%             end
%         end
        mu = 0;
        sigma = 1;
        pd = makedist('Normal','mu',mu,'sigma',sigma);
        x = contexts{armOptimal};
        normalizedX = (beta - dot(theta,x))/sqrt(dot(phi,x)+L);
        accumOptimalRewards = dot(theta,x)*(cdf(pd,normalizedX))^(-1);
        
        individualRegret = accumOptimalRewards - accumObservedRewards;
        accumuRegrets = [accumuRegrets;individualRegret];
    end
    results{repeat} = accumuRegrets;
end

meanRegret = 0;
for i = 1:total_repeat
    meanRegret = meanRegret + cumsum(results{i});
end
meanRegret = meanRegret/total_repeat;
save('K=2.5,beta=-3,5.mat')
