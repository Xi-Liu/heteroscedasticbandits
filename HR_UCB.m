function armChosen = HR_UCB(beta, xi, d, lambda, L, contexts, sContexts, sRewards)

mu = 0;
sigma = 1;
pd = makedist('Normal','mu',mu,'sigma',sigma);

V = sContexts' * sContexts + lambda * eye(d);

thetaHat = inv(V) * (sContexts') * sRewards;

varEpsilons = [];

for i = 1: length(sContexts)
    x = sContexts(i,:);
    varEpsilons = [varEpsilons; (sRewards(i) - thetaHat' * (x'))^2 - L];
end

phiHat = inv(V) * (sContexts') * varEpsilons;

indexes = [];
for i = 1 : length(contexts)
    x = contexts{i};
    u = thetaHat' * (x');
    v = phiHat' * (x');
    normalizedX = (beta - dot(thetaHat,x))/sqrt(dot(phiHat,x)+L);
    part1 = u * (cdf(pd,normalizedX))^(-1);
    part3 = sqrt(x * inv(V) * x');
    indexes = [indexes; part1 + xi * part3];
end
[M,armChosen] = max(indexes);

end