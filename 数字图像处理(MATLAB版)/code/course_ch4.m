function course_ch4(n)
% 第四章 图像复原
    % n=1   imnoise 随机噪声
    % n=2   imnoise2 随机噪声
    % n=3   imnoise3 周期噪声
    % n=4   估计噪声参数
    % n=5   仅有噪声的复原 去噪
    % n=6   退化函数模型
    % n=7   带有卷积的复原 去卷积

    if(n == 1)
        % imnoise 随机噪声
        f = imread("./pic/Lena.jpg");
        f = rgb2gray(f);
        f = tofloat(f);
        g1 = imnoise(f, 'gaussian', 0, 0.01);   % [均值, 方差]
        g2 = imnoise(f, 'poisson');

        fig_plot(f, g1, "gaussian");
        fig_plot(f, g2, "poisson");

    elseif(n == 2)
        % imnoise2 随机噪声
        f = imread("./pic/Lena.jpg");
        f = rgb2gray(f);
        f = tofloat(f);
        [M, N] = size(f);
        noise1 = imnoise2('gaussian', M, N, 0, 0.1); % [均值, 标准差]
        g1 = noise1 + f;
        noise2 = imnoise2('uniform', M, N);
        g2 = noise2 + f;
        noise3 = imnoise2('rayleigh', M, N);
        g3 = 0.25*noise3 + f;

        fig_plot(f, g1, "gaussian");
        fig_plot(f, g2, "uniform");
        fig_plot(f, g3, "rayleigh");
    
    elseif(n == 3)
        % imnoise3 周期噪声
        f = imread("./pic/Lena.jpg");
        f = rgb2gray(f);
        f = tofloat(f);
        [M, N] = size(f);
        C = [2 4; 30 -6];
        [n, N, S] = imnoise3(M, N, C);
        g = n + f;
        figure;imshow(S, []),title("加噪图对数频谱")
        [x, y] = find(S>0.5);
        fig_plot(f, g, "imnois3");

    elseif(n == 4)
        % 估计噪声参数
        f = imread('Fig0404(a).tif');
        [B, c, r] = roipoly(f);         % 获得roi区域
        [p, npix] = histroi(f, c, r);   % 计算roi直方图
        [v, unv] = statmoments(p, 2);   % 计算中心距
        % u = unv(1); sigma = unv(2);
        X = imnoise2('gaussian', npix, 1, 147, 20);
        figure;
        subplot(121),bar(p,1);title('噪音roi区域直方图');
        subplot(122),hist(X, 130);title('gaussian噪音');
    elseif(n == 5)
        % 仅有噪声的复原(空间滤波) g = f + n  空间噪声滤波
        f = imread('Fig0219(a).tif');
        [M, N] = size(f);

        % 加噪
        R1 = imnoise2('salt & pepper', M, N, 0.1, 0);
        g1 = f;
        g1(R1==0) = 0;   % 仅被胡椒噪声污染的图像 黑点
        R2 = imnoise2('salt & pepper', M, N, 0, 0.1);
        g2 = f;
        g2(R2==0) = 255; % 仅被盐粒噪声污染的图像 白点
        figure;
        subplot(131),imshow(f),title('原图');
        subplot(132),imshow(g1,[]),title('仅被胡椒噪声污染');
        subplot(133),imshow(g2,[]),title('仅被盐粒噪声污染');

        % 反调和均值滤波去噪
        f1 = spfilt(g1, 'chmean', 3, 3, 1.5);   % 去除黑点Q>0
        f2 = spfilt(g2, 'chmean', 3, 3, -1.5);  % 去除白点Q<0
        figure;
        subplot(121),imshow(g1,[]),title('反调和均值去除胡椒噪声');
        subplot(122),imshow(g2,[]),title('反调和均值去除盐粒噪声');

        % 最大最小值滤波去噪
        f1 = spfilt(g1, 'max', 3, 3);
        f2 = spfilt(g2, 'min', 3, 3);
        figure;
        subplot(121),imshow(g1,[]),title('最大值去除胡椒噪声');
        subplot(122),imshow(g2,[]),title('最小值去除盐粒噪声');

        % 自适应空间滤波器
        %   加噪
        g = imnoise(f, 'salt & pepper', 0.25);
        f1 = medfilt2(g, [7 7], 'symmetric');
        f2 = adpmedian(g, 7);
        figure;
        subplot(121),imshow(f1),title('中值去除椒盐噪声');
        subplot(122),imshow(f2),title('自适应均值去除椒盐噪声');

    elseif(n == 6)
        % 退化函数模型——g = H(f) + n = h ** f + n;
        f = checkerboard(8);
        [M, N] = size(f);
        h = fspecial('motion', 7, 45);
        gb = imfilter(f, h, 'circular');
        n = imnoise2('gaussian', M, N, 0, sqrt(0.001));
        g = gb + n;
        figure;
        subplot(221),imshow(pixeldup(f, 8), []),title('原图');
        subplot(222),imshow(pixeldup(gb, 8), []),title('原图卷积');
        subplot(223),imshow(pixeldup(g, 8), []),title('带噪图');
        subplot(224),imshow(pixeldup(n, 8), []),title('噪声');

    elseif(n == 7)
        % 直接逆滤波 维纳滤波
        f = checkerboard(8);
        [M, N] = size(f);
        h = fspecial('motion', 7, 45);  % 滤波器

        n = imnoise2('gaussian', M, N, 0, sqrt(0.001)); % 噪声
        Sn = abs(fft2(n)).^2;
        nA = sum(Sn(:)) / (M*N);
        Sf = abs(fft2(f)).^2;
        fA = sum(Sf(:)) / (M*N);
        R = nA / fA;                    % 噪信功率比

        gb = imfilter(f, h, 'circular');% 卷积
        g = gb + n;                     % 加噪图
        f1 = deconvwnr(g, h);           % 直接逆滤波
        f2 = deconvwnr(g, h, R);        % 维纳滤波
        NCORR = fftshift(real(ifft2(Sn)));
        ICORR = fftshift(real(ifft2(Sf)));
        f3 = deconvwnr(g, h, NCORR, ICORR); % 自相关逆滤波
        figure;
        subplot(221),imshow(pixeldup(g, 8)),title('带噪图');
        subplot(222),imshow(pixeldup(f1, 8)),title('直接逆滤波');
        subplot(223),imshow(pixeldup(f2, 8)),title('噪信功率比逆滤波');
        subplot(224),imshow(pixeldup(f3, 8)),title('自相关逆滤波');

    end
end

function fig_plot(f, g, type)
    n = g - f;
    figure;
    subplot(221),imshow(f),title('原图');
    subplot(222),imshow(g),title('加噪');
    subplot(223),imshow(n, []),title(type+"噪声");
    subplot(224),hist(n, 50),title('噪声直方图');
end