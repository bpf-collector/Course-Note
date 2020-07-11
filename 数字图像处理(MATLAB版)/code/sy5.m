function sy5(n)
% ʵ��5 �ռ��˲�
    % n=1   laplacian�޸�a
    % n=2   laplacian�޸�ģ��
    % n=3   ��ֵ�˲�
    % n=4   ���ֵ�˲�

if(n == 1)
    % laplacian�޸�a
    f = imread('Fig0217(a).tif');   %���򱱼�ͼ��
    w1 = fspecial('laplacian', 0);
    g1 = imfilter(f, w1, 'replicate');
    w2 = fspecial('laplacian', 0.5);
    g2 = imfilter(f, w2, 'replicate');
    w3 = fspecial('laplacian', 1);
    g3 = imfilter(f, w3, 'replicate');

    figure;
    subplot(221), imshow(f), title('ԭͼ');
    subplot(222), imshow(f-g1), title('laplacian a=0');
    subplot(223), imshow(f-g2), title('laplacian a=0.5');
    subplot(224), imshow(f-g3), title('laplacian a=1');

elseif(n == 2)
    % laplacian�޸�ģ��
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
    subplot(231), imshow(f), title('ԭͼ');
    subplot(232), imshow(f-g1), title('laplacian 3\times3');
    subplot(233), imshow(f-g2), title('laplacian 5\times5');
    subplot(234), imshow(f-g3), title('laplacian 9\times9');
    subplot(235), imshow(f-g4), title('laplacian 15\times15');

elseif(n == 3)
    % ��ֵ�˲�
    f = imread('Fig0219(a).tif');
    % f = im2single(f);
    fn = imnoise(f, 'salt & pepper', 0.2);  % ����

    fprintf("(1)�˲�ǰ: %f\n", psnr(f, fn))

    g1 = medfilt2(fn, 'symmetric'); % ��һ����ֵ�˲�
    fprintf("(2)��һ���˲���: %f\n", psnr(f, g1))

    g2 = medfilt2(g1, 'symmetric'); % �ڶ�����ֵ�˲�
    fprintf("(3)�ڶ����˲���: %f\n", psnr(f, g2))

    g3 = medfilt2(g2, 'symmetric'); % ��������ֵ�˲�
    fprintf("(4)�������˲���: %f\n", psnr(f, g3))
    figure;
    subplot(231), imshow(f), title('ԭͼ')
    subplot(232);imshow(fn-f, []),title('����');
    subplot(233), imshow(fn), title('ԭͼ����')
    subplot(234), imshow(g1), title('��һ���˲�')
    subplot(235), imshow(g2), title('�ڶ����˲�')
    subplot(236), imshow(g3), title('�������˲�')

elseif(n == 4)
    % ���ֵ�˲�
    f = imread('Fig0219(a).tif');
    [M, N] = size(f);
    fn = imnoise(f, 'salt & pepper', 0.2);  % ����

    fprintf("(1)�˲�ǰ: %f\n", psnr(f, fn))

    g1 = ordfilt2(fn, M*N, ones(M, N)); % ��һ�����ֵ�˲�
    fprintf("(2)��һ���˲���: %f\n", psnr(f, g1))

    g2 = ordfilt2(g1, M*N, ones(M, N)); % �ڶ������ֵ�˲�
    fprintf("(3)�ڶ����˲���: %f\n", psnr(f, g2))

    g3 = ordfilt2(g2, M*N, ones(M, N)); % ���������ֵ�˲�
    fprintf("(4)�������˲���: %f\n", psnr(f, g3))
    figure;
    subplot(231), imshow(f), title('ԭͼ')
    subplot(232);imshow(fn-f, []),title('����');
    subplot(233), imshow(fn), title('ԭͼ����')
    subplot(234), imshow(g1), title('��һ���˲�')
    subplot(235), imshow(g2), title('�ڶ����˲�')
    subplot(236), imshow(g3), title('�������˲�')
    
end
end
