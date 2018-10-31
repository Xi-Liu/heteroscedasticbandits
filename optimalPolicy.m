function armOptimal = optimalPolicy(beta, theta, phi, L, contexts)

mu = 0;
sigma = 1;
pd = makedist('Normal','mu',mu,'sigma',sigma);

indexes = [];
for i = 1:length(contexts)
    x = contexts{i};
    normalizedX = (beta - dot(theta,x))/sqrt(dot(phi,x)+L);
    indexes = [indexes;dot(theta,x)*(cdf(pd,normalizedX))^(-1)];
end
[M,armOptimal] = max(indexes);
end

