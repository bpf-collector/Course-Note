function sy9(n)
% 实验九 彩色图像
    % n=1   生成红色圆
    % n=2   生产四方块rgb图
    % n=3   对角线三颜色图 rgb图像方法
    % n=4   对角线三颜色图 索引图像方法
    % n=5   边缘检测、图像分割

if(n == 1)
    % 生成红色圆
    a = zeros(256);
    for i=1:256
        for j=1:256
            if(sqrt((i-128)*(i-128)+(j-128)*(j-128)) <= 128)
                a(i,j) = 1;
            end
        end
    end
    b = zeros(256);
    f = cat(3, a, b, b); % red
    figure;imshow(f)

elseif(n == 2)
    % 生产四方块rgb图
    M = 512;
    a = getOnePiece(M, 1, 256, 1, 256);
    b = getOnePiece(M, 257, 512, 1, 256);
    c = getOnePiece(M, 1, 256, 257, 512);
    f = cat(3, a, b, c);
    imshow(f);

elseif(n == 3)
    % 对角线三颜色图 rgb图像方法
    a = ones(12);
    b = ones(12);
    c = ones(12);
    for i = 1:11
        a(i, i+1) = 0;
        a(i+1, i) = 0;
        b(i, i) = 0;
        b(i, i+1) = 0;
        c(i, i) = 0;
        c(i+1, i) = 0;
    end
    b(12,12) = 0;
    c(12,12) = 0;
    f = cat(3, a,b,c);
    figure;imshow(f);

elseif(n == 4)
    % 对角线三颜色图 索引图像方法
    x = ones(12);
    map = [
        1 1 1;
        1 0 0;
        0 1 0;
        0 0 1];
    for i = 1:11
        x(i, i) = 2;
        x(i, i+1) = 4;
        x(i+1, i) = 3;
    end
    x(12,12)=2;
    figure,imshow(x, map);

elseif(n == 5)
    % 生成图像 ============================================
    C = [180 257; 280 187; 280 327];
    a = pane(512, C(1,:), 90);
    b = pane(512, C(2,:), 90);
    c = pane(512, C(3,:), 90);
    f = cat(3, a, b, c);
    figure;
    subplot(221),imshow(a),title("R");
    subplot(222),imshow(b),title("G");
    subplot(223),imshow(c),title("B");
    subplot(224),imshow(f),title("f");
    
    % 边缘检测 ===========================================
    fr = f(:, :, 1);
    fg = f(:, :, 2);
    fb = f(:, :, 3);

    w1 = [-1 -2 -1; 0 0 0; 1 2 1];      % sobel模板，可用于求梯度
    frgdx = imfilter(fr, w1, 'replicate');
    frgdy = imfilter(fr, w1', 'replicate');
    fggdx = imfilter(fg, w1, 'replicate');
    fggdy = imfilter(fg, w1', 'replicate');
    fbgdx = imfilter(fb, w1, 'replicate');
    fbgdy = imfilter(fb, w1', 'replicate');
    frgd = abs(frgdx) + abs(frgdy);
    fggd = abs(fggdx) + abs(fggdy);
    fbgd = abs(fbgdx) + abs(fbgdy);
    fg =  abs(frgd) + abs(fggd) + abs(fbgd);    % 梯度合成
    figure;
    subplot(221),imshow(frgd),title('frgd2');
    subplot(222),imshow(fggd),title('fggd2');
    subplot(223),imshow(fbgd),title('fbgd2');
    subplot(224),imshow(fg),title('fg');

    [VG, A, PRG] = colorgrad(f);    % 向量梯度
    figure;
    subplot(121),imshow(VG),title("向量梯度VG");
    subplot(122),imshow(PRG),title("向量梯度PRG");

    w2 = [1 1 1; 1 -8 1; 1 1 1];
    g2 = imfilter(f, w2, 'replicate');
    figure;imshow(g2),title("拉普拉斯滤波");

    % 图像分割 ===========================================
    figure; mask = roipoly(f);          % 获取roi区域
    [M, N, K] = size(f);
    g = cat(3, immultiply(mask, fr), immultiply(mask, fg), immultiply(mask, fb));
    I = reshape(g, M*N, K);
    idx = find(mask);
    I = double((I(idx, :)));
    [C, m] = covmatrix(I);      % 协方差矩阵、均值  
    sd = sqrt(diag(C));         % 标准差
    T = max(ceil(sd));          % 阈值
    figure;
    subplot(121),imshow(f),title('原图');
    subplot(122),imshow(g),title('roi区域');
    % ====== 欧式距离
    T1 = colorseg('euclidean', f, 1*T, m);
    T2 = colorseg('euclidean', f, 2*T, m);
    figure;
    subplot(121),imshow(T1),title('欧式 1T');
    subplot(122),imshow(T2),title('欧式 2T');
    % ====== 马氏距离
    T1 = colorseg('mahalanobis', f, 1*T, m, C);
    T2 = colorseg('mahalanobis', f, 2*T, m, C);
    figure;
    subplot(121),imshow(T1),title('马氏 1T');
    subplot(122),imshow(T2),title('马氏 2T');
    % ====== 马氏距离
    T1 = colorseg('mahalanobis', f, 1*T, m);
    T2 = colorseg('mahalanobis', f, 2*T, m);
    figure;
    subplot(121),imshow(T1),title('马氏2 1T');
    subplot(122),imshow(T2),title('马氏2 2T');
end
end

function h = pane(M, Co, R)
    h = zeros(M);
    for i=1:M
        for j=1:M
            if( (i-Co(1))^2+(j-Co(2))^2 <= R^2)
                h(i, j) = 1;
            end
        end
    end
end

function h = getOnePiece(M, rols, role, cols, cole)
% 得到四方块的其中一个分量
    h = zeros(M);
    h(rols:role, cols:cole) = 1;
end