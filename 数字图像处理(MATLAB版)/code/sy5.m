function sy5(n)
% 实验5 空间滤波
    % n=1   laplacian修改a
    % n=2   laplacian修改模板
    % n=3   中值滤波
    % n=4   最大值滤波

if(n == 1)
    % laplacian修改a
    f = imread('Fig0217(a).tif');   %月球北极图像
    w1 = fspecial('laplacian', 0);
    g1 = imfilter(f, w1, 'replicate');
    w2 = fspecial('laplacian', 0.5);
    g2 = imfilter(f, w2, 'replicate');
    w3 = fspecial('laplacian', 1);
    g3 = imfilter(f, w3, 'replicate');

    figure;
    subplot(221), imshow(f), title('原图');
    subplot(222), imshow(f-g1), title('laplacian a=0');
    subplot(223), imshow(f-g2), title('laplacian a=0.5');
    subplot(224), imshow(f-g3), title('laplacian a=1');

elseif(n == 2)
    % laplacian修改模板
    f = imread('Fig0217(a).tif');
    w1 = LaplacianMask(3);
    w2 = LaplacianMask(5);
    w3 = LaplacianMask(9);
    w4 = LaplacianMask(15);
    g1 = imfilter(f, w1, 'replicate');
    g2 = imfilter(f, w2, 'replicate');
    g3 = imfilter(f, w3, 'replicate');
    g4 = imfilter(f, w4, 'replicate');

    figure;
    subplot(231), imshow(f), title('原图');
    subplot(232), imshow(f-g1), title('laplacian 3\times3');
    subplot(233), imshow(f-g2), title('laplacian 5\times5');
    subplot(234), imshow(f-g3), title('laplacian 9\times9');
    subplot(235), imshow(f-g4), title('laplacian 15\times15');

elseif(n == 3)
    % 中值滤波
    f = imread('Fig0219(a).tif');
    % f = im2single(f);
    fn = imnoise(f, 'salt & pepper', 0.2);  % 加噪

    fprintf("(1)滤波前: %f\n", psnr(f, fn))

    g1 = medfilt2(fn, 'symmetric'); % 第一次中值滤波
    fprintf("(2)第一次滤波后: %f\n", psnr(f, g1))

    g2 = medfilt2(g1, 'symmetric'); % 第二次中值滤波
    fprintf("(3)第二次滤波后: %f\n", psnr(f, g2))

    g3 = medfilt2(g2, 'symmetric'); % 第三次中值滤波
    fprintf("(4)第三次滤波后: %f\n", psnr(f, g3))
    figure;
    subplot(231), imshow(f), title('原图')
    subplot(232);imshow(fn-f, []),title('噪声');
    subplot(233), imshow(fn), title('原图加噪')
    subplot(234), imshow(g1), title('第一次滤波')
    subplot(235), imshow(g2), title('第二次滤波')
    subplot(236), imshow(g3), title('第三次滤波')

elseif(n == 4)
    % 最大值滤波
    f = imread('Fig0219(a).tif');
    [M, N] = size(f);
    fn = imnoise(f, 'salt & pepper', 0.2);  % 加噪

    fprintf("(1)滤波前: %f\n", psnr(f, fn))

    g1 = ordfilt2(fn, M*N, ones(M, N)); % 第一次最大值滤波
    fprintf("(2)第一次滤波后: %f\n", psnr(f, g1))

    g2 = ordfilt2(g1, M*N, ones(M, N)); % 第二次最大值滤波
    fprintf("(3)第二次滤波后: %f\n", psnr(f, g2))

    g3 = ordfilt2(g2, M*N, ones(M, N)); % 第三次最大值滤波
    fprintf("(4)第三次滤波后: %f\n", psnr(f, g3))
    figure;
    subplot(231), imshow(f), title('原图')
    subplot(232);imshow(fn-f, []),title('噪声');
    subplot(233), imshow(fn), title('原图加噪')
    subplot(234), imshow(g1), title('第一次滤波')
    subplot(235), imshow(g2), title('第二次滤波')
    subplot(236), imshow(g3), title('第三次滤波')
    
end
end
