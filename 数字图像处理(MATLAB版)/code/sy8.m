function sy8(n)
% 实验八 图像复原
    % n=1   去噪：随机噪声
    % n=2   去噪：周期噪声
    % n=3   去卷积

if(n == 1)
    % 去噪：随机噪声
    f = imread('./pic/Lena_gray_512.tif');
    f = tofloat(f);
    [M, N] = size(f);
    n = imnoise2('erlang', M, N);           % 爱尔兰噪声
    g = f + 0.03*n;
    % figure;
    % subplot(121),imshow(f),title('原图');
    % subplot(122),imshow(g),title('加噪图');

    data = load('sy8_1.mat');
    B = data.B;
    c = data.c;
    r = data.r;
    % [B, c, r] = roipoly(g);                 % 选择roi区域
    [h, npix] = histroi(g, c, r);           % 计算roi直方图
    [v, unv] = statmoments(h, 2);           % 计算mu, sigma
    figure;
    subplot(121),imshow(1-B),title('roi区域');
    subplot(122);bar(h, 1),title('roi直方图');

    g1 = spfilt(g, 'amean', 3, 3);
    g2 = spfilt(g, 'gmean', 3, 3);
    g3 = spfilt(g, 'hmean', 3, 3);
    g4 = spfilt(g, 'chmean', 3, 3, 0.5);
    g5 = spfilt(g, 'median', 3, 3);
    g6 = spfilt(g, 'max', 3, 3);
    g7 = spfilt(g, 'min', 3, 3);
    g8 = spfilt(g, 'midpoint', 3, 3);
    g9 = spfilt(g, 'atrimmed', 3, 3, 2);

    figure;
    subplot(331),imshow(g1),title('amean 3*3');
    subplot(332),imshow(g2),title('gmean 3*3');
    subplot(333),imshow(g3),title('hmean 3*3');
    subplot(334),imshow(g4),title('chmean 3*3 Q=0.5');
    subplot(335),imshow(g5),title('median 3*3');
    subplot(336),imshow(g6),title('max 3*3');
    subplot(337),imshow(g7),title('min 3*3');
    subplot(338),imshow(g8),title('midpoint 3*3');
    subplot(339),imshow(g9),title('atrimmed 3*3 D=2');

    figure;
    subplot(331),imhist(g1),title('amean 3*3');
    subplot(332),imhist(g2),title('gmean 3*3');
    subplot(333),imhist(g3),title('hmean 3*3');
    subplot(334),imhist(g4),title('chmean 3*3 Q=0.5');
    subplot(335),imhist(g5),title('median 3*3');
    subplot(336),imhist(g6),title('max 3*3');
    subplot(337),imhist(g7),title('min 3*3');
    subplot(338),imhist(g8),title('midpoint 3*3');
    subplot(339),imhist(g9),title('atrimmed 3*3 D=2');

elseif(n == 2)
    % 去噪：周期噪声
    f = imread('./pic/Lena_gray_512.tif');
    [f, revertclass] = tofloat(f);
    [M, N] = size(f);

    C = [0 32; 0 64; 16 16; 32 0; 64 0; -16 16];
    A = [0.1  0.3  0.9  0.5  0.01  0.2];
    n = imnoise3(M, N, C, A);           % 生成周期噪声
    g = n + f;                          % 加噪图
    G = fft2(g);
    Gc = fftshift(log(1+abs(G)));
    figure;
    subplot(121),imshow(g),title('加噪图');
    subplot(122),imshow(Gc, []),title('加噪图对数频谱');

    % 带阻滤波
    D0 = [32, 64, 22.6];
    H = bandfilter2('gaussian', 'reject', M, N, D0, 2, 0);
    g1 = dftfilt(g, H);
    G1 = fft2(g1);
    G1c = fftshift(log(1+abs(G1)));
    figure;
    subplot(121),imshow(g1),title('bandfilter2滤波');
    subplot(122),imshow(G1c, []),title('bandfilter2滤波对数频谱');

    % 陷波带阻滤波
    C1 = [256 288; 256 320; 272 272; 288 256; 320 256; 240 272];
    H = cnotch('gaussian', 'reject', M, N, C1, 2);
    g2 = dftfilt(g, H);
    g2 = revertclass(g2);
    G2 = fft2(g2);
    G2c = fftshift(log(1+abs(G2)));
    figure;
    subplot(121),imshow(g2),title('cnotch滤波');
    subplot(122),imshow(G2c, []),title('cnotch滤波对数频谱');
elseif(n == 3)
    % 去卷积
    f1 = checkerboard();
    % figure;imshow(f1);
    % figure;imshow(pixeldup(f1, 4));

    f2 = imread('cameraman.tif');
    % figure;imshow(f);
    % figure;imshow(pixeldup(f2, 4));
    h1 = fspecial('motion', 7, 45);
    h2 = fspecial('motion', 7, -45);
    g1 = imfilter(f1, h1, 'circular');
    g2 = imfilter(f2, h1, 'circular');
    g3 = imfilter(f1, h2, 'circular');
    g4 = imfilter(f2, h2, 'circular');
    % figure;
    % subplot(321),imshow(pixeldup(f1, 1)),title('f1');
    % subplot(323),imshow(pixeldup(g1, 1)),title('motion 45');
    % subplot(325),imshow(pixeldup(g3, 1)),title('motion -45');
    % subplot(322),imshow(pixeldup(f2, 1)),title('f2');
    % subplot(324),imshow(pixeldup(g2, 1)),title('motion 45');
    % subplot(326),imshow(pixeldup(g4, 1)),title('motion -45');

    n1 = imnoise2('gaussian',size(f1,1),size(f1,2),0,sqrt(0.001));
    % figure;imshow(pixeldup(g1, 6),[]),title('f1卷积');
    g1 = g1 + n1;            % g1 = h1**f1 + n1;
    % figure;imshow(pixeldup(g1, 6),[]),title('f1卷积加噪');
    N1 = fft2(n1);
    % figure;imshow(pixeldup(fftshift(log(1+abs(N1))), 6),[]),title('f1的噪声频谱');

    n2 = imnoise2('gaussian',size(f2,1),size(f2,2),0,sqrt(0.001));
    % figure;imshow(pixeldup(g2, 2),[]),title('f2卷积');
    g2 = tofloat(g2) + n2;   % g2 = h1**f2 + n2;
    % figure;imshow(pixeldup(g2, 2),[]),title('f2卷积加噪');
    N2 = fft2(n2);
    % figure;imshow(pixeldup(fftshift(log(1+abs(N2))), 2),[]),title('f2的噪声频谱');

    R1 = computeR(f1, n1, "f1 ");
    R2 = computeR(f2, n2, "f2 ");
    f1_1 = deconvwnr(g1, h1, 0);       % 直接逆滤波
    f1_2 = deconvwnr(g1, h1, R1);      % 维纳滤波
    figure;
    subplot(121),imshow(pixeldup(f1_1, 4),[]),title("f1 直接逆滤波");
    subplot(122),imshow(pixeldup(f1_2, 4),[]),title("f1 R维纳滤波");
    f1_1 = deconvwnr(g1, h1, 0.1);       % 直接逆滤波
    f1_2 = deconvwnr(g1, h1, 0.5);      % 维纳滤波
    figure;
    subplot(121),imshow(pixeldup(f1_1, 4),[]),title("f1 0.1维纳滤波");
    subplot(122),imshow(pixeldup(f1_2, 4),[]),title("f1 0.5维纳滤波");
    f1_1 = deconvwnr(g1, h1, 1);       % 直接逆滤波
    f1_2 = deconvwnr(g1, h1, 1.5);      % 维纳滤波
    figure;
    subplot(121),imshow(pixeldup(f1_1, 4),[]),title("f1 1.0维纳滤波");
    subplot(122),imshow(pixeldup(f1_2, 4),[]),title("f1 1.5维纳滤波");

    f2_1 = deconvwnr(g2, h2, 0);       % 直接逆滤波
    f2_2 = deconvwnr(g2, h2, R2);      % 维纳滤波
    figure;
    subplot(121),imshow(pixeldup(f2_1, 4),[]),title("f2 直接逆滤波");
    subplot(122),imshow(pixeldup(f2_2, 4),[]),title("f2 R维纳滤波");
end
end

function R = computeR(f, noise, plot)
    % 计算噪性功率比
    Sn = abs(fft2(noise)).^2;
    nA = sum(Sn(:)) / numel(noise);

    Sf = abs(fft2(f)).^2;
    fA = sum(Sf(:)) / numel(f);

    R = nA / fA;
    if(plot ~= "")
        figure;
        subplot(121),imshow(fftshift(Sn),[]),title(plot+'Sn');
        subplot(122),imshow(fftshift(Sf),[]),title(plot+'Sf');
    end
end