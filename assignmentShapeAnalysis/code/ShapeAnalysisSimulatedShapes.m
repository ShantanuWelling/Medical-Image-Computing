mat = load("ellipses2D.mat");

mat
numpointsets = mat.numOfPointSets;
numpoints = mat.numOfPoints;
data = mat.pointSets;

cmap = jet(numpointsets);
scatter(reshape(data(1,:,:),numpoints,numpointsets),reshape(data(2,:,:),numpoints,numpointsets),5,cmap);

data2 = data;

for i = 1:numpointsets
    avg = sum(data2(1,:,i))/numpoints;
    data2(1,:,i) = data2(1,:,i) - avg;
    avg2 = sum(data2(2,:,i))/numpoints;
    data2(2,:,i) = data2(2,:,i) - avg2;
end

for i = 1:numpointsets
    s1 = sum(data2(1,:,i).*data2(1,:,i));
    s2 = sum(data2(2,:,i).*data2(2,:,i));
    data2(:,:,i) = data2(:,:,i)/sqrt(s1 + s2);
end

[mean1, ~] = code11(data2,numpointsets);
[mean2, ~] = code22(data,numpointsets,numpoints);

for k = 1:10
    [mean1, data2] = code11(data2,numpointsets);
end

plot(mean1(1,:),mean1(2,:),'--k*');
hold on;
scatter1 = scatter(reshape(data2(1,:,:),numpoints,numpointsets),reshape(data2(2,:,:),numpoints,numpointsets),5,cmap);
alpha(scatter1,0.2);
hold off;

for k = 1:10
    [mean2, data] = code22(data,numpointsets,numpoints);
end

plot(mean2(1,:),mean2(2,:),'--k*');
hold on;
scatter1 = scatter(reshape(data(1,:,:),numpoints,numpointsets),reshape(data(2,:,:),numpoints,numpointsets),5,cmap);
alpha(scatter1,0.2);
hold off;

datan = reshape(data - mean2, [2*numpoints, numpointsets]);
data2n = reshape(data2 - mean1, [2*numpoints, numpointsets]);

cov_mat1 = zeros(2 * numpoints,2 * numpoints);
cov_mat2 = zeros(2 * numpoints,2 * numpoints);
for i=1:2*numpoints
    for j=1:2*numpoints
        cov_mat1(i,j) = data2n(i,:)*data2n(j,:).';
        cov_mat2(i,j) = datan(i,:)*datan(j,:).';
    end
end

cov_mat1 = cov_mat1 / numpointsets;
cov_mat2 = cov_mat2 / numpointsets;

[V1,D1] = eig(cov_mat1);
[V2,D2] = eig(cov_mat2);

plot(D1);
plot(D2);

for i = 1:3
    figure;
    plot(mean1(1,:),mean1(2,:),'--r');
    hold on;
    max1 = sqrt(D1(2 * numpoints - i + 1,2 * numpoints - i + 1));
    vec1 = reshape(V1(:,2 * numpoints - i + 1),2,numpoints);
    vec11 = vec1(1,:);
    vec11 = vec11 / norm(vec11);
    vec12 = vec1(2,:);
    vec12 = vec12 / norm(vec12);
    plot(mean1(1,:) + 3*max1*vec11, mean1(2,:) + 3*max1*vec12,'--b');
    plot(mean1(1,:) - 3*max1*vec11, mean1(2,:) - 3*max1*vec12,'--g');
    legend('Mean', 'Mean + 3 S.D.', 'Mean - 3 S.D.');
    hold off;
end

for i = 1:3
    figure;
    plot(mean2(1,:),mean2(2,:),'--r');
    hold on;
    max1 = sqrt(D2(2 * numpoints - i + 1,2 * numpoints - i + 1));
    vec2 = reshape(V2(:,2 * numpoints - i + 1),2,numpoints);
    vec21 = vec2(1,:);
    vec21 = vec21 / norm(vec21);
    vec22 = vec2(2,:);
    vec22 = vec22 / norm(vec22);
    plot(mean2(1,:) + 3*max1*vec21, mean2(2,:) + 3*max1*vec22,'--b');
    plot(mean2(1,:) - 3*max1*vec21, mean2(2,:) - 3*max1*vec22,'--g');
    legend('Mean', 'Mean + 3 S.D.', 'Mean - 3 S.D.');
    hold off;
end

function [ rot ] = code1(set1, set2)
    x = set1';
    y = set2';
    t = x*y.';
    [U, ~, V] = svd(t);
    if(det(V*U') ~= 1)
        d = size(U);
        w = eye(d(1));
        w(d(1),d(1)) = -1;
        rot = V*w*U';
    else
        rot = V*U';
    end
end

function [ rot, s, t ] = code2(set1, set2,N)
    x1 = sum(set1(1,:));
    x2 = sum(set2(1,:));
    y1 = sum(set1(2,:));
    y2 = sum(set2(2,:));
    W = N;
    Z = sum(set2(1,:).*set2(1,:)) + sum(set2(2,:).*set2(2,:));
    C1 = sum(set1(1,:).*set2(1,:)) + sum(set1(2,:).*set2(2,:));
    C2 = sum(set1(2,:).*set2(1,:)) - sum(set1(1,:).*set2(2,:));
    mat1 = [x2, -y2, W, 0; y2, x2, 0, W; Z, 0, x2, y2; 0, Z, -y2, x2];
    mat2 = [x1, y1, C1, C2];
    mat3 = mat1 \ mat2';
    ax = mat3(1,1);
    ay = mat3(2,1);
    tx = mat3(3,1);
    ty = mat3(4,1);
    theta = atan2(ay,ax);
    rot = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    t = [tx, ty].';
    s = sqrt(ax*ax + ay*ay);
end

function [ mean, newData ] = code11(pointsets,P)
    mean = sum(pointsets, 3)/P;
    mean = mean/sqrt(sum(mean(1,:).*mean(1,:))+ sum(mean(2,:).*mean(2,:)));
    for j = 1:P
        rot = code1(pointsets(:,:,j),mean);
        pointsets(:,:,j) = (rot*pointsets(:,:,j)')';
    end
    newData = pointsets;
end

function [ mean, newData ] = code22(pointsets,P,N)
    mean = sum(pointsets, 3)/P;
    mean = mean/sqrt(sum(mean(1,:).*mean(1,:))+ sum(mean(2,:).*mean(2,:)));
    for j = 1:P
        [rot, s, t] = code2(mean,pointsets(:,:,j),N);
        pointsets(:,:,j) = s*rot*pointsets(:,:,j) + t;
    end
    newData = pointsets;
end
