function H = bandfilter3(type, M, N, Radii, Width, Order)
%巴特沃斯带阻、带通滤波器
%   TYPE: 'reject'(带阻)、'pass'(带通)
%   M, N: the number of rows and columns in the filter
%   Radii: 表示每个频带中心的D0，1*k数组
%   Width: 表示每个频带的宽度，1*k数组
%   Order: 表示每个频带的阶数，1*k数组

    [U, V] = dftuv(M, N);
    D = hypot(U, V);
    k = numel(Radii);       % 频带个数

    if(k == 1)          % 一个频带
        H = btwReject(D, Radii, Width, Order);  % 带阻滤波器

    elseif(k == 2)      % 两个频带
        if(numel(Order) == 1)
            Order(2) = Order;       % 改为1*k数组
        end
        h1 = btwReject(D, Radii(1), Width(1), Order(1));  % 带阻滤波器h1
        h2 = btwReject(D, Radii(2), Width(2), Order(2));  % 带阻滤波器h2
        H = add_bandfilter(h1, h2);                       % 滤波器相加
    end

    if strcmp(type, 'pass')
        % 带通滤波器
        H = 1 - H;
    end
end

function H = add_bandfilter(h1, h2)
% 两个带通滤波器相加, 返回带通滤波器
    %   h1, h2为带通滤波器
    H = h1 + h2;
    H = H - min(H(:));
    H = H / max(H(:));
end

function H = btwReject(D, D0, W, n)
% 带阻滤波器
    H = 1 ./ (1 + (((D*W)./(D.^2-D0^2 + eps)).^(2*n)));
end
