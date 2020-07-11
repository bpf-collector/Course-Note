function course_ch3(n)
% 第三章 频域处理
    % n=1   二维离散傅里叶变换
    % n=2   频域滤波
    % n=3   空间域滤波 => 频率域滤波
    % n=4   理想低通频率滤波器
    % n=5   巴特沃斯低通频率滤波器
    % n=6   高斯低通滤波器
    % n=7   线框图与表面图
    % n=8   低通滤波器
    % n=9   高通滤波器
    % n=10  高通滤波
    % n=11  高频强调滤波
    % n=12  陷波带阻滤波器cnotch函数
    % n=13  陷波带阻滤波器recnotch函数

if(n == 1)
    % 二维离散傅里叶变换
    % ==============================原图
    f = imread('Fig0303(a).tif');
    % f = im2single(f);
    F = fft2(f);            % DFT
    f_ = ifft2(F);          % IDFT
    f_ = real(f_);          % 除去误差
    Fc = fftshift(F);       % 移动频率中心
    SF = abs(Fc);           % 原图频谱图
    SF2 = log(1 + abs(Fc)); % 对数变换频谱图
    fphi = angle(Fc);       % 相角
    fphi = atan2(imag(F), real(F));
    [M, N] = size(f);
    % fprintf("频率中心：(%d, %d)\n", floor(M/2)+1, floor(N/2)+1);
    % ==============================平移
    g = zeros(512);
    g(209+50:305+50, 249-80:265-80) = f(209:305, 249:265);
    % g = im2single(g);
    G = fft2(g);            % DFT
    Gc = fftshift(G);       % 移动频率中心
    SG = abs(Gc);           % 原图频谱图
    SG2 = log(1 + abs(Gc)); % 对数变换频谱图
    gphi = angle(Gc);       % 相角
    % ==============================旋转
    h = imrotate(f, -45, 'bilinear', 'crop');
    H = fft2(h);            % DFT
    Hc = fftshift(H);       % 移动频率中心
    SH = abs(Hc);           % 原图频谱图
    SH2 = log(1 + abs(Hc)); % 对数变换频谱图
    hphi = angle(Hc);       % 相角
    %?==============================画图
    figure;
    subplot(331), imshow(f), title('原图');
    subplot(332), imshow(SF2, []), title('对数变换频谱图');
    subplot(333), imshow(fphi, []), title('原图相角');
    subplot(334), imshow(g), title('平移原图');
    subplot(335), imshow(SG2, []), title('对数变换频谱图');
    subplot(336), imshow(gphi, []), title('平移相角');
    subplot(337), imshow(h), title('旋转原图');
    subplot(338), imshow(SH2, []), title('对数变换频谱图');
    subplot(339), imshow(hphi, []), title('旋转相角');

elseif(n == 2)
    % 频域滤波
    f = imread('Fig0305(a).tif');       % 读取图像
    figure,
    subplot(221),imshow(f),title('原图');

    % 无填充效果
    f = imread('Fig0305(a).tif');       % 读取图像
    [M, N] = size(f);                   % 图像大小
    [f, revertclass] = tofloat(f);      % 转为float类型
    F = fft2(f);                        % DFT
    H = lpfilter('gaussian', M, N, 10); % 脉冲响应的傅里叶变换(传递函数) 10表示标准差
    G = H.*F;                           % 频率域滤波
    g = ifft2(G);                       % DFT反变换
    gc = revertclass(g);                % 转回unit8类型
    subplot(222),imshow(gc),title('无填充效果');

    % 填充效果 点乘方法
    f = imread('Fig0305(a).tif');       % 读取图像
    [f, revertclass] = tofloat(f);      % 转为float类型
    PQ = paddedsize(size(f));           % 计算填充大小
    F = fft2(f, PQ(1), PQ(2));          % DFT
    H = lpfilter('gaussian', PQ(1), PQ(2), 2*10);
    G = H.*F;                           % 频率域滤波
    g = ifft2(G);                       % DFT反变换
    gc = g(1:size(f,1), 1:size(f,2));   % 截取大小
    gc = revertclass(gc);               % 转回unit8类型
    subplot(223),imshow(gc),title('填充效果（频率域滤波）');

    % 卷积方法(与填充效果一致)
    f = imread('Fig0305(a).tif');       % 读取图像
    [f, revertclass] = tofloat(f);      % 转为float类型
    h = fspecial('gaussian', 15, 7);    % 生成高斯模板
    g = imfilter(f, h);                 % 滤波器
    g = revertclass(g);                 % 转回unit8类型
    subplot(224), imshow(g), title('卷积（空间域滤波）');

elseif(n == 3)
    % 空间域滤波 => 频率域滤波
    f = imread('Fig0309(a).tif');
    f = tofloat(f);
    F = fft2(f);                    % DFT
    Fc = fftshift(log(1+abs(F)));   % 频率中心转到左上角
    PQ = paddedsize(size(f));       % 计算需要填充的数值
    h = fspecial('sobel')';         % 已旋转180度
    H = freqz2(h, PQ(1), PQ(2));    % DFT
    H = ifftshift(H);               % 频率中心移到左上角
    gs = imfilter(f, h);            % 空间域滤波
    gf = dftfilt(f, H);             % 频率域滤波

    figure;
    subplot(221),imshow(f),title('原图');
    subplot(222),imshow(Fc, []),title('对数处理的频谱');
    subplot(223),imshow(abs(gs), []),title('空间域滤波');
    subplot(224),imshow(abs(gf), []),title('频率域滤波');

elseif(n == 4)
    % 理想低通频率滤波器
    f = imread('Fig0313(a).tif');
    [f, revertclass] = tofloat(f);
    PQ = paddedsize(size(f));
    [U, V] = dftuv(PQ(1), PQ(2));
    D = hypot(U, V);
    D0 = 0.05*PQ(2);            % 定义截止频率
    F = fft2(f, PQ(1), PQ(2));
    H = single(D <= D0);        % 理想滤波器
    g = dftfilt(f, H);
    g = revertclass(g);
    
    figure;
    subplot(221),imshow(f),title('原图');
    subplot(222),imshow(log(1+abs(fftshift(F))), []),title('原图对数频谱');
    subplot(223),imshow(g),title('滤波');
    subplot(224),imshow(fftshift(H)),title('理想低通滤波器');

elseif(n == 5)
    % 巴特沃斯低通频率滤波器
    f = imread('Fig0313(a).tif');
    [f, revertclass] = tofloat(f);
    PQ = paddedsize(size(f));
    [U, V] = dftuv(PQ(1), PQ(2));
    D = hypot(U, V);
    D0 = 0.05*PQ(2);            % 定义截止频率
    F = fft2(f, PQ(1), PQ(2));
    n = 2;
    H = 1./(1+(D./D0).^(2*n));  % 巴特沃斯低通滤波器
    g = dftfilt(f, H);
    g = revertclass(g);
    
    figure;
    subplot(221),imshow(f),title('原图');
    subplot(222),imshow(log(1+abs(fftshift(F))), []),title('原图对数频谱');
    subplot(223),imshow(g),title('滤波');
    subplot(224),imshow(fftshift(H)),title('巴特沃斯低通滤波器');

elseif(n == 6)
    % 高斯低通滤波器
    f = imread('Fig0313(a).tif');
    [f, revertclass] = tofloat(f);
    PQ = paddedsize(size(f));
    [U, V] = dftuv(PQ(1), PQ(2));
    D = hypot(U, V);
    D0 = 0.05*PQ(2);            % 定义截止频率
    F = fft2(f, PQ(1), PQ(2));
    H = exp(-D.^2/(2*(D0^2)));  % 高斯低通滤波器
    g = dftfilt(f, H);
    g = revertclass(g);
    
    figure;
    subplot(221),imshow(f),title('原图');
    subplot(222),imshow(log(1+abs(fftshift(F))), []),title('原图对数频谱');
    subplot(223),imshow(g),title('滤波');
    subplot(224),imshow(fftshift(H)),title('高斯低通滤波器');

elseif(n == 7)
    % 线框图与表面图
    H = fftshift(lpfilter('gaussian', 500, 500, 50));
    figure;
    mesh(double(H(1:10:500, 1:10:500)));        % 线框图
    axis tight, axis off, colormap([0 0 0]);
    figure;
    surf(double(H(1:10:500, 1:10:500)));        % 表面图
    axis tight, axis off, colormap(gray);
    shading interp;

elseif(n == 8)
    %  低通滤波器
    f = imread('Fig0313(a).tif');
    PQ = paddedsize(size(f));
    D0 = 0.05*PQ(2);
    H1 = lpfilter('ideal', PQ(1), PQ(2), D0);
    H2 = lpfilter('btw', PQ(1), PQ(2), D0);
    H3 = lpfilter('gaussian', PQ(1), PQ(2), D0);
    g1 = dftfilt(f, H1);
    g2 = dftfilt(f, H2);
    g3 = dftfilt(f, H3);

    figure;
    subplot(221),imshow(f),title('原图');
    subplot(222),imshow(g1),title('ideal滤波');
    subplot(223),imshow(g2),title('btw滤波');
    subplot(224),imshow(g3),title('gaussian滤波');

elseif(n == 9)
    % 高通滤波器
    H1 = fftshift(hpfilter('ideal', 500, 500, 50));      % 理想高通滤波器
    H2 = fftshift(hpfilter('btw', 500, 500, 50, 2));    % 巴特沃斯高通滤波器
    H3 = fftshift(hpfilter('gaussian', 500, 500, 50));  % 高斯高通滤波器
    figure;
    subplot(231),imshow(H1, []), title('理想高通滤波器 2D');
    subplot(234),mesh(double(H1(1:10:end, 1:10:end))),title('ideal D0=50'),axis off;
    subplot(232),imshow(H2, []), title('巴特沃斯高通滤波器 2D');
    subplot(235),mesh(double(H2(1:10:end, 1:10:end))),title('btw D0=50 n=2'),axis off;
    subplot(233),imshow(H3, []), title('高斯高通滤波器 2D');
    subplot(236),mesh(double(H3(1:10:end, 1:10:end))),title('gaussian D0=50'),axis off;
    
elseif(n == 10)
    % 高通滤波
    f = imread('Fig0313(a).tif');
    PQ = paddedsize(size(f));
    f = tofloat(f);
    D0 = 0.05 * PQ(1);
    H = hpfilter('gaussian', PQ(1), PQ(2), D0);
    g = dftfilt(f, H);
    figure;
    subplot(121),imshow(f),title('原图');
    subplot(122),imshow(g),title('滤波后');

elseif(n == 11)
    % 高频强调滤波
    f = imread('Fig0319(a).tif');
    PQ = paddedsize(size(f));
    D0 = 0.05 * PQ(1);
    HBW = hpfilter('btw', PQ(1), PQ(2), D0, 2); % btw滤波器
    H = 0.5 + 2*HBW;                            % 高频强调滤波器
    g1 = dftfilt(f, HBW, 'fltpoint');
    g1 = gscale(g1);                % 拉伸图像元素大小到[0, 255]
    g2 = dftfilt(f, H, 'fltpoint'); % float类型
    g2 = gscale(g2);                % 拉伸图像元素大小到[0, 255]
    g2_eq = histeq(g2, 256);        % 对g2进行直方图均衡
    g3 = dftfilt(f, H);             % uint8类型
    g3_eq = histeq(g3, 256);        % 对g3进行直方图均衡

    figure;
    subplot(221),imshow(f),title('原图');
    subplot(222),imhist(f);title('原图直方图');
    subplot(223),imshow(g1),title('btw滤波');
    subplot(224),imhist(g1);title('btw滤波直方图');
    figure;
    subplot(221),imshow(g2),title('g2 高频滤波');
    subplot(222),imhist(g2);title('g2 高频滤波直方图');
    subplot(223),imshow(g2_eq),title('g2 高频滤波 直方图均衡后');
    subplot(224),imhist(g2_eq);title('g2 高频滤波 直方图均衡');
    figure;
    subplot(221),imshow(g3),title('g3 高频滤波');
    subplot(222),imhist(g3);title('g3 高频滤波直方图');
    subplot(223),imshow(g3_eq),title('g3 高频滤波 直方图均衡后');
    subplot(224),imhist(g3_eq);title('g3 高频滤波 直方图均衡');

elseif(n == 12)
    % 陷波带阻滤波器cnotch函数
    f = imread('Fig0321(a).tif');
    [f, revertclass] = tofloat(f);
    [M, N] = size(f);
    F = fft2(f);                        % DFT
    Fc = log(1 + abs(fftshift(F)));                     % 对数频谱
    S = gscale(Fc);     % imtool(S)函数获得尖峰坐标
    C1 = [99 154; 128 163];
    H1 = cnotch('gaussian', 'reject', M, N, C1, 5);     % 陷波带阻滤波器
    p1 = gscale(fftshift(H1).*(tofloat(S)));
    g1 = dftfilt(f, H1);                                % 陷波带阻C1 频域滤波
    g1 = revertclass(g1);
    figure;
    subplot(121),imshow(f),title('原图');
    subplot(122),imshow(S),title('原图对数频谱');
    figure;
    subplot(121),imshow(p1),title('陷波带阻滤波 ');
    subplot(122),imshow(g1),title('陷波带阻滤波 ');

    C2 = [99 154; 128 163; 49 160; 133 233; 55 132; 108 225; 112 74];
    H2 = cnotch('gaussian', 'reject', M, N, C2, 5);     % 陷波带阻滤波器
    p2 = gscale(fftshift(H2).*(tofloat(S)));
    g2 = dftfilt(f, H2);                                % 陷波带阻C2 频域滤波
    g2 = revertclass(g2);
    figure;
    subplot(121),imshow(p2),title('陷波带阻滤波 ');
    subplot(122),imshow(g2),title('陷波带阻滤波 ');

elseif(n == 13)
    %陷波带阻滤波器recnotch函数
    f = imread('Fig0322(a).tif');
    [f, revertclass] = tofloat(f);
    [M, N] = size(f);
    F = fft2(f);
    Fc = log(1 + abs(fftshift(F)));
    S = gscale(Fc);
    HR = recnotch('reject', 'vertical', M, N, 3, 15, 15);    % 陷波带阻滤波器
    g1 = dftfilt(f, HR);            % 频域滤波 去掉空间干扰 即去噪
    g1 = revertclass(g1);
    HP = recnotch('pass', 'vertical', M, N, 3, 15, 15);      % 陷波带通滤波器
    g2 = dftfilt(f, HP);            % 频域滤波 得到空间干扰 即噪音
    g2 = gscale(g2);

    figure;
    subplot(121),imshow(f),title('原图');
    subplot(122),imshow(S),title('原图对数频谱');
    figure;
    subplot(121),imshow(HR),title('陷波带阻滤波器');
    subplot(122),imshow(g1),title('陷波带阻滤波 去噪');
    figure;
    subplot(121),imshow(HP),title('陷波带通滤波器');
    subplot(122),imshow(g2),title('陷波带通滤波 噪音');

end