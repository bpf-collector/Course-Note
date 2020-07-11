function H = bandfilter2(type, band, M, N, D0, W, n)
    [U, V] = dftuv(M, N);
    D = hypot(U, V);        % 生成频率网格点
    H = zeros(M, N);        % 距离
    k = numel(D0);          % 频带个数

    if numel(W) == 1
        W = W * ones(1, k);
    end
    if numel(n) == 1
        n = n * ones(1, k);
    end

    switch lower(type)
    case 'ideal'
        fun = @idealReject;
    case 'btw'
        fun = @btwReject;
    case 'gaussian'
        fun = @gaussianReject;
    end

    for i=1:k
        p = fun(D, D0(i), W(i), n(i));
        H = H + p;
    end

    H = H - min(H(:));
    H = H / max(H(:));

    if strcmp(band, 'pass')
        H = 1 - H;
    end

end

function H = btwReject(D, D0, W, n)
% 带阻滤波器
    H = 1 ./ (1 + (((D*W)./(D.^2-D0^2 + eps)).^(2*n)));
end

function H = gaussianReject(D, D0, W, n)
% 高斯带阻滤波器
    H = 1 - exp(-((D.^2 - D0^2)./(D.*W + eps)).^2);
end

function H = idealReject(D, D0, W, n)
    RI = D <= D0 - (W/2);   % Points of region inside the inner 
                            % boundary of the reject band are labeled 1.
                            % All other points are labeled 0.
                             
    RO = D >= D0 + (W/2);   % Points of region outside the outer 
                            % boundary of the reject band are labeled 1.
                            % All other points are labeled 0.
                            
    H = tofloat(RO | RI);   % Ideal bandreject filter.
end