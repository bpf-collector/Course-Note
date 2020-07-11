function course_ch5(n)
% 第五章 彩色图像处理
    % n=1   RGB图像
    % n=2   索引图像
    % n=3   索引图像
    % n=4   预定义的彩色映射
    % n=5   处理RGB与索引图像
    % n=6   彩色图像平滑处理
    % n=7   彩色图像锐化处理
    % n=8   梯度检测彩色边缘
    % n=9   图像分割

    if(n == 1)
        % RGB图像
        rgbcube();
    elseif(n == 2)
        % 索引图像
        load mandrill;
        figure;imshow(X, map),title('索引图像');
    elseif(n == 3)
        % 索引图像
        X = zeros(200);
        X(:, 1:50) = 1;
        X(:, 51:100) = 64;
        X(:, 101:150) = 128;
        X(:, 151:200) = 256;
        map = [0 0 0; 1 1 1];
        figure,
        subplot(121),imshow(X, map),title('map1');
        map(64, :) = [1 0 0];
        map(128, :) = [0 1 0];
        map(256, :) = [0 0 1];
        subplot(122),imshow(X, map),title('map2');
    elseif(n == 4)
        % 预定义的彩色映射
        load mandrill;
        colormap(copper);
        subplot(121), imshow(X, map),title('索引图像');
        subplot(122), imshow(X, copper),title('预定义的彩色映射');
    elseif(n == 5)
        % 处理RGB与索引图像
        f = imread('Fig0604(a).tif');
        [X1, map1] = rgb2ind(f, 8, 'nodither');
        %  8 表示处理后图像只有8种颜色
        [X2, map2] = rgb2ind(f, 8, 'dither');
        figure;
        subplot(131),imshow(f),title('原图');
        subplot(132),imshow(X1, map1),title('非抖动处理');
        subplot(133),imshow(X2, map2),title('抖动处理');
        g = rgb2gray(f);
        g1 = dither(g);
        figure;
        subplot(121),imshow(g),title('灰度图像');
        subplot(122),imshow(g1),title('抖动处理');
    elseif(n == 6)
        % 彩色图像平滑处理
        f = imread('Fig0622(a).tif');
        [f, revertclass] = tofloat(f);
        w = fspecial('average', 25);
        % ======================================= rgb图像平滑处理
        f1 = imfilter(f, w, 'replicate'); % 作用相同于对分量分别滤波
        f1 = revertclass(f1);
        fR = f(:,:,1);
        fG = f(:,:,2);
        fB = f(:,:,3);
        fR_filtered = imfilter(fR, w, 'replicate');
        fG_filtered = imfilter(fG, w, 'replicate');
        fB_filtered = imfilter(fB, w, 'replicate');
        f2 = cat(3, fR_filtered, fG_filtered, fB_filtered);
        f2 = revertclass(f2);
        figure;
        subplot(121),imshow(f),title('原图');
        subplot(122),imshow(f1),title('平滑处理');
        % ======================================= hsi图像平滑处理
        f_hsi = rgb2hsi(f);
        H = f_hsi(:, :, 1);
        S = f_hsi(:, :, 2);
        I = f_hsi(:, :, 3);
        H_filtered = imfilter(H, w, 'replicate');
        S_filtered = imfilter(S, w, 'replicate');
        I_filtered = imfilter(I, w, 'replicate');
        f1 = cat(3, H, S, I_filtered);
        f1 = hsi2rgb(f1);
        f1 = revertclass(f1);
        f2 = cat(3, H_filtered, S_filtered, I_filtered);
        f2 = hsi2rgb(f2);
        f2 = revertclass(f2);
        figure;
        subplot(131),imshow(f),title('原图');
        subplot(132),imshow(f1),title('滤波I分量');
        subplot(133),imshow(f2),title('滤波HSI分量');
    elseif(n == 7)
        % 彩色图像锐化处理
        f = imread('Fig0622(a).tif');
        [f, revertclass] = tofloat(f);
        w = fspecial('average', 5);
        f1 = imfilter(f, w, 'replicate');
        lapmask = [1 1 1; 1 -8 1; 1 1 1];
        f2 = f1 - imfilter(f1, lapmask, 'replicate');
        figure;
        subplot(131),imshow(f),title('原图');
        subplot(132),imshow(f1),title('滤波图');
        subplot(133),imshow(f2),title('锐化图');
    elseif(n == 8)
        % 梯度检测彩色边缘
        fr = imread('Fig0627(a).tif');
        fg = imread('Fig0627(b).tif');
        fb = imread('Fig0627(c).tif');
        f = cat(3, fr, fg, fb);
        figure;
        subplot(221),imshow(fr),title('R');
        subplot(222),imshow(fg),title('G');
        subplot(223),imshow(fb),title('B');
        subplot(224),imshow(f),title('f');
        f = tofloat(f);
        [VG, A, PRG] = colorgrad(f);
        err = abs(VG - PRG);
        figure;
        subplot(131),imshow(VG),title('向量梯度');
        subplot(132),imshow(PRG),title('分量梯度求和的图像');
        subplot(133),imshow(err,[]),title('VG-PRG');
    elseif(n == 9)
        % 图像分割
        f = imread('Fig0630(a).tif');
        mask = roipoly(f);          % 选取roi区域
        fr = immultiply(mask, f(:,:,1));
        fg = immultiply(mask, f(:,:,2));
        fb = immultiply(mask, f(:,:,3));
        g = cat(3, fr, fg, fb);
        [M, N, K] = size(g);
        I = reshape(g, M*N, K);     % K=3
        idx = find(mask);
        I = double(I(idx, :));
        [C, m] = covmatrix(I);      % 计算协方差矩阵C和均值m
        d = diag(C);                % 方差d为位于协方差矩阵C的对角线上
        sd = sqrt(d);               % 标准差sd
        T = max(ceil(sd));          % 阈值为标准差最大值的向上取整
        figure;
        subplot(121),imshow(f),title('原图');
        subplot(122),imshow(g),title('roi区域');
        % ================================= 欧式距离算法
        T12 = colorseg('euclidean', f, 1*T, m);
        T24 = colorseg('euclidean', f, 2*T, m);
        T48 = colorseg('euclidean', f, 4*T, m);
        T96 = colorseg('euclidean', f, 8*T, m);
        figure;
        subplot(221),imshow(T12),title('欧式 T=12');
        subplot(222),imshow(T24),title('欧式 T=24');
        subplot(223),imshow(T48),title('欧式 T=48');
        subplot(224),imshow(T96),title('欧式 T=96');
        % ================================= 马式距离算法
        T12 = colorseg('mahalanobis', f, 1*T, m, C);
        T24 = colorseg('mahalanobis', f, 2*T, m, C);
        T48 = colorseg('mahalanobis', f, 4*T, m, C);
        T96 = colorseg('mahalanobis', f, 8*T, m, C);
        figure;
        subplot(221),imshow(T12),title('马式 T=12');
        subplot(222),imshow(T24),title('马式 T=24');
        subplot(223),imshow(T48),title('马式 T=48');
        subplot(224),imshow(T96),title('马式 T=96');
    end
end