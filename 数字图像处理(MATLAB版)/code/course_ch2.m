function course_ch2(n)
%第二章 灰度变换与空间滤波
    % n=1   imadjust灰度变换
    % n=2   对数变换
    % n=3   多种绘图方式
    % n=4   直方图与归一化直方图
    % n=5   直方图均衡
    % n=6   直方图匹配（规定化直方图）
    % n=7   线性空间滤波
    % n=8   非线性空间滤波
    % n=9   线性空间滤波器
    % n=10  非线性空间滤波器
    
if(n == 1)
    % imadjust灰度变换
    f = imread('Fig0203(a).tif');
    g1 = imadjust(f, [0 1], [1 0]);     % 负片
    g2 = imcomplement(f);               % 负片
    g3 = imadjust(f, [0.5 0.75], []);
    g4 = imadjust(f, [], [], 2);        % gamma>1效果会变暗
    g5 = imadjust(f, stretchlim(f), [1 0]);
    figure,
    subplot(231),imshow(f), title('原图');
    subplot(232),imshow(g1), title('imadjust负片');
    subplot(233),imshow(g2), title('imcomplement负片');
    subplot(234),imshow(g3), title('[0.75 0.75] => [0 1]');
    subplot(235),imshow(g4), title('gamma=2');
    subplot(236),imshow(g5), title('stretchlim => [1 0]');
elseif(n == 2)
    % 对数变换
    f = imread('Fig0205(a).tif');
    g = im2uint8(mat2gray(log(1+double(f))));
    figure;
    subplot(121),imshow(f),title('频谱[0:1.5\times10^6]');
    subplot(122),imshow(g),title('对数变换后频谱[0:14]');

elseif(n == 11)
    % 对比度拉伸变换
    f = imread('Fig0206(a).tif');
    g = intrans(f, 'stretch', mean2(tofloat(f)), 0.9);
    figure;
    subplot(121),imshow(f),title('原图');
    subplot(122),imshow(g),title('对比度拉伸');

elseif(n == 3)
    % 直方图与归一化直方图
    f = imread('Fig0203(a).tif');
    p = imhist(f)/numel(f);
    figure;
    subplot(131),imshow(f),title('原图');
    subplot(132),imhist(f);title('直方图');
    subplot(133),plot(p),title('归一化直方图');

elseif(n == 4)
    % 多种绘图方式
    f = imread('Fig0203(a).tif');
    h = imhist(f, 25);
    horz = linspace(0, 255, 25);
    fhandle = @tanh;

    figure;
    subplot(131),bar(horz, h),title('条形图');
    set(gca, 'xtick', 0:50:255);
    subplot(132),stem(horz, h, 'fill'),title('杆状图');
    set(gca, 'xtick', 0:50:255);
    subplot(133),fplot(fhandle, [-2, 2], 'r:p'), title('tanh(x)');

elseif(n == 5)
    % 直方图均衡
    f = imread('Fig0208(a).tif');
    g = histeq(f, 256); % 直方图均衡
    gam = intrans(f, 'gamma', 0.4);

    figure;
    subplot(231),imshow(f),title('原图');
    subplot(234),imhist(f);title('原图直方图'),ylim('auto');
    subplot(232),imshow(g),title('直方图均衡');
    subplot(235),imhist(g);title("直方图均衡后的直方图"),ylim('auto');
    subplot(233),imshow(gam),title('gamma变换');
    subplot(236),imhist(gam);title("gamma的直方图 gamma=0.4"),ylim('auto');

elseif(n == 6)
    % 直方图匹配（规定化直方图）
    f = imread('Fig0210(a).tif');
    h = histeq(f, 256);     % 直方图均衡
    p = manualhist;
    g1 = histeq(f, p);      % 自定义函数直方图匹配
    % 工具箱函数直方图匹配
    g2 = adapthisteq(f, 'NumTiles', [25 25], 'ClipLimit', 0.05);

    figure;
    subplot(421),imshow(f),title('原图');
    subplot(422),stem(imhist(f));title('直方图'),ylim('auto');
    subplot(423),imshow(h),title('直方图均衡');
    subplot(424),stem(imhist(h));title('直方图'),ylim('auto');
    subplot(425),imshow(g1),title('manualhist直方图匹配');
    subplot(426),stem(imhist(g1));title('直方图'),ylim('auto');
    subplot(427),imshow(g2),title('adapthisteq直方图匹配');
    subplot(428),stem(imhist(g2));title('直方图'),ylim('auto');

elseif(n == 7)
    % 线性空间滤波
    f = imread('Fig0216(a).tif');
    f = im2single(f);
    w = ones(31);
    g1 = imfilter(f, w);                % 0填充
    g2 = imfilter(f, w, 'replicate');   % 复制边界填充
    g3 = imfilter(f, w, 'circular');    % 周期边界填充
    f = im2uint8(f);
    g4 = imfilter(f, w, 'replicate');   % 不转浮点型出现截断
    w = 1 / 31^2 .* w;                  % 滤波器归一化
    g5 = imfilter(f, w, 'replicate');   % 归一化滤波器不会截断
    figure;
    subplot(231),imshow(f),title('原图');
    subplot(232),imshow(g1, []),title('0填充 single');
    subplot(233),imshow(g2, []),title('复制边界填充 single');
    subplot(234),imshow(g3, []),title('周期边界填充 single');
    subplot(235),imshow(g4, []),title('复制边界填充 uint8');
    subplot(236),imshow(g5, []),title('复制边界填充 uint8 归一化滤波器');

elseif(n == 8)
    % 非线性空间滤波
    f = imread('Fig0216(a).tif');
    fp = padarray(f, [3 3], 'replicate');   % 边界填充
    gmean = @(A) prod(A, 1).^(1/size(A, 1));% 求几何平均
    g = colfilt(fp, [3 3], 'sliding', gmean);%非线性空间滤波
    [M, N] = size(f);
    g = g((1:M)+3, (1:N)+3);                % 去掉填充部分
    figure;
    subplot(121),imshow(f),title('原图');
    subplot(122),imshow(g),title('非线性空间滤波');

elseif(n == 9)
    % 线性空间滤波器
    f = imread('./pic/Lena.jpg');
    f = im2single(rgb2gray(f));         % 转为浮点型
    w1 = fspecial('average', 3);
    w2 = fspecial('average', 5);
    w3 = fspecial('average', 7);
    g1 = imfilter(f, w1, 'replicate');
    g2 = imfilter(f, w2, 'replicate');
    g3 = imfilter(f, w3, 'replicate');

    w4 = fspecial('gaussian', [3 5], 0.5);% 高斯滤波器
    g4 = imfilter(f, w4, 'replicate');    % 高斯平滑滤波后
    fd = f - g4;        % unsharp滤波器
    g5 = f + 1.5*fd;    % 高频强调图 详细化
    g6 = f + 0.7*fd;    % 低频弱化图 模糊化

    figure;
    subplot(221),imshow(f),title('原图');
    subplot(222),imshow(g1),title('均值平滑3*3卷积核');
    subplot(223),imshow(g2),title('均值平滑5*5卷积核');
    subplot(224),imshow(g3),title('均值平滑7*7卷积核');
    figure;
    subplot(221),imshow(g3),title('高斯平滑');
    subplot(222),imshow(fd, []),title('unsharp滤波器');
    subplot(223),imshow(g5),title('高频强调图');
    subplot(224),imshow(g6),title('低频弱化图');

    f = imread('Fig0217(a).tif');
    f = tofloat(f);
    w1 = fspecial('laplacian', 0);
    w2 = [1 1 1; 1 -8 1; 1 1 1];
    g1 = imfilter(f, w1, 'replicate');  % 拉普拉斯滤波后
    g3 = f - g1;                        % 高频强调图
    g2 = imfilter(f, w2, 'replicate');  % 拉普拉斯滤波后
    g4 = f - g2;                        % 高频强调图
    figure;
    subplot(131),imshow(f),title('月球北极');
    subplot(132),imshow(g3, []),title('-4拉普拉斯滤波高频强调图')
    subplot(133),imshow(g4, []),title('-8拉普拉斯滤波高频强调图')

elseif(n == 10)
    % 非线性空间滤波器
    f = imread('Fig0219(a).tif');
    fn = imnoise(f, 'salt & pepper', 0.2);  % 添加椒盐噪声
    g1 = medfilt2(fn);              % 中值滤波 0填充
    g2 = medfilt2(fn, 'symmetric'); % 中值滤波 边界对称填充

    figure;
    subplot(221),imshow(f),title('原图');
    subplot(222),imshow(fn),title('带椒盐噪声');
    subplot(223),imshow(g1),title('中值滤波 0填充');
    subplot(224),imshow(g2),title('中值滤波 边界对称填充');

end