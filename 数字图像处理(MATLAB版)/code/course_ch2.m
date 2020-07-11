function course_ch2(n)
%�ڶ��� �Ҷȱ任��ռ��˲�
    % n=1   imadjust�Ҷȱ任
    % n=2   �����任
    % n=3   ���ֻ�ͼ��ʽ
    % n=4   ֱ��ͼ���һ��ֱ��ͼ
    % n=5   ֱ��ͼ����
    % n=6   ֱ��ͼƥ�䣨�涨��ֱ��ͼ��
    % n=7   ���Կռ��˲�
    % n=8   �����Կռ��˲�
    % n=9   ���Կռ��˲���
    % n=10  �����Կռ��˲���
    
if(n == 1)
    % imadjust�Ҷȱ任
    f = imread('Fig0203(a).tif');
    g1 = imadjust(f, [0 1], [1 0]);     % ��Ƭ
    g2 = imcomplement(f);               % ��Ƭ
    g3 = imadjust(f, [0.5 0.75], []);
    g4 = imadjust(f, [], [], 2);        % gamma>1Ч����䰵
    g5 = imadjust(f, stretchlim(f), [1 0]);
    figure,
    subplot(231),imshow(f), title('ԭͼ');
    subplot(232),imshow(g1), title('imadjust��Ƭ');
    subplot(233),imshow(g2), title('imcomplement��Ƭ');
    subplot(234),imshow(g3), title('[0.75 0.75] => [0 1]');
    subplot(235),imshow(g4), title('gamma=2');
    subplot(236),imshow(g5), title('stretchlim => [1 0]');
elseif(n == 2)
    % �����任
    f = imread('Fig0205(a).tif');
    g = im2uint8(mat2gray(log(1+double(f))));
    figure;
    subplot(121),imshow(f),title('Ƶ��[0:1.5\times10^6]');
    subplot(122),imshow(g),title('�����任��Ƶ��[0:14]');

elseif(n == 11)
    % �Աȶ�����任
    f = imread('Fig0206(a).tif');
    g = intrans(f, 'stretch', mean2(tofloat(f)), 0.9);
    figure;
    subplot(121),imshow(f),title('ԭͼ');
    subplot(122),imshow(g),title('�Աȶ�����');

elseif(n == 3)
    % ֱ��ͼ���һ��ֱ��ͼ
    f = imread('Fig0203(a).tif');
    p = imhist(f)/numel(f);
    figure;
    subplot(131),imshow(f),title('ԭͼ');
    subplot(132),imhist(f);title('ֱ��ͼ');
    subplot(133),plot(p),title('��һ��ֱ��ͼ');

elseif(n == 4)
    % ���ֻ�ͼ��ʽ
    f = imread('Fig0203(a).tif');
    h = imhist(f, 25);
    horz = linspace(0, 255, 25);
    fhandle = @tanh;

    figure;
    subplot(131),bar(horz, h),title('����ͼ');
    set(gca, 'xtick', 0:50:255);
    subplot(132),stem(horz, h, 'fill'),title('��״ͼ');
    set(gca, 'xtick', 0:50:255);
    subplot(133),fplot(fhandle, [-2, 2], 'r:p'), title('tanh(x)');

elseif(n == 5)
    % ֱ��ͼ����
    f = imread('Fig0208(a).tif');
    g = histeq(f, 256); % ֱ��ͼ����
    gam = intrans(f, 'gamma', 0.4);

    figure;
    subplot(231),imshow(f),title('ԭͼ');
    subplot(234),imhist(f);title('ԭͼֱ��ͼ'),ylim('auto');
    subplot(232),imshow(g),title('ֱ��ͼ����');
    subplot(235),imhist(g);title("ֱ��ͼ������ֱ��ͼ"),ylim('auto');
    subplot(233),imshow(gam),title('gamma�任');
    subplot(236),imhist(gam);title("gamma��ֱ��ͼ gamma=0.4"),ylim('auto');

elseif(n == 6)
    % ֱ��ͼƥ�䣨�涨��ֱ��ͼ��
    f = imread('Fig0210(a).tif');
    h = histeq(f, 256);     % ֱ��ͼ����
    p = manualhist;
    g1 = histeq(f, p);      % �Զ��庯��ֱ��ͼƥ��
    % �����亯��ֱ��ͼƥ��
    g2 = adapthisteq(f, 'NumTiles', [25 25], 'ClipLimit', 0.05);

    figure;
    subplot(421),imshow(f),title('ԭͼ');
    subplot(422),stem(imhist(f));title('ֱ��ͼ'),ylim('auto');
    subplot(423),imshow(h),title('ֱ��ͼ����');
    subplot(424),stem(imhist(h));title('ֱ��ͼ'),ylim('auto');
    subplot(425),imshow(g1),title('manualhistֱ��ͼƥ��');
    subplot(426),stem(imhist(g1));title('ֱ��ͼ'),ylim('auto');
    subplot(427),imshow(g2),title('adapthisteqֱ��ͼƥ��');
    subplot(428),stem(imhist(g2));title('ֱ��ͼ'),ylim('auto');

elseif(n == 7)
    % ���Կռ��˲�
    f = imread('Fig0216(a).tif');
    f = im2single(f);
    w = ones(31);
    g1 = imfilter(f, w);                % 0���
    g2 = imfilter(f, w, 'replicate');   % ���Ʊ߽����
    g3 = imfilter(f, w, 'circular');    % ���ڱ߽����
    f = im2uint8(f);
    g4 = imfilter(f, w, 'replicate');   % ��ת�����ͳ��ֽض�
    w = 1 / 31^2 .* w;                  % �˲�����һ��
    g5 = imfilter(f, w, 'replicate');   % ��һ���˲�������ض�
    figure;
    subplot(231),imshow(f),title('ԭͼ');
    subplot(232),imshow(g1, []),title('0��� single');
    subplot(233),imshow(g2, []),title('���Ʊ߽���� single');
    subplot(234),imshow(g3, []),title('���ڱ߽���� single');
    subplot(235),imshow(g4, []),title('���Ʊ߽���� uint8');
    subplot(236),imshow(g5, []),title('���Ʊ߽���� uint8 ��һ���˲���');

elseif(n == 8)
    % �����Կռ��˲�
    f = imread('Fig0216(a).tif');
    fp = padarray(f, [3 3], 'replicate');   % �߽����
    gmean = @(A) prod(A, 1).^(1/size(A, 1));% �󼸺�ƽ��
    g = colfilt(fp, [3 3], 'sliding', gmean);%�����Կռ��˲�
    [M, N] = size(f);
    g = g((1:M)+3, (1:N)+3);                % ȥ����䲿��
    figure;
    subplot(121),imshow(f),title('ԭͼ');
    subplot(122),imshow(g),title('�����Կռ��˲�');

elseif(n == 9)
    % ���Կռ��˲���
    f = imread('./pic/Lena.jpg');
    f = im2single(rgb2gray(f));         % תΪ������
    w1 = fspecial('average', 3);
    w2 = fspecial('average', 5);
    w3 = fspecial('average', 7);
    g1 = imfilter(f, w1, 'replicate');
    g2 = imfilter(f, w2, 'replicate');
    g3 = imfilter(f, w3, 'replicate');

    w4 = fspecial('gaussian', [3 5], 0.5);% ��˹�˲���
    g4 = imfilter(f, w4, 'replicate');    % ��˹ƽ���˲���
    fd = f - g4;        % unsharp�˲���
    g5 = f + 1.5*fd;    % ��Ƶǿ��ͼ ��ϸ��
    g6 = f + 0.7*fd;    % ��Ƶ����ͼ ģ����

    figure;
    subplot(221),imshow(f),title('ԭͼ');
    subplot(222),imshow(g1),title('��ֵƽ��3*3�����');
    subplot(223),imshow(g2),title('��ֵƽ��5*5�����');
    subplot(224),imshow(g3),title('��ֵƽ��7*7�����');
    figure;
    subplot(221),imshow(g3),title('��˹ƽ��');
    subplot(222),imshow(fd, []),title('unsharp�˲���');
    subplot(223),imshow(g5),title('��Ƶǿ��ͼ');
    subplot(224),imshow(g6),title('��Ƶ����ͼ');

    f = imread('Fig0217(a).tif');
    f = tofloat(f);
    w1 = fspecial('laplacian', 0);
    w2 = [1 1 1; 1 -8 1; 1 1 1];
    g1 = imfilter(f, w1, 'replicate');  % ������˹�˲���
    g3 = f - g1;                        % ��Ƶǿ��ͼ
    g2 = imfilter(f, w2, 'replicate');  % ������˹�˲���
    g4 = f - g2;                        % ��Ƶǿ��ͼ
    figure;
    subplot(131),imshow(f),title('���򱱼�');
    subplot(132),imshow(g3, []),title('-4������˹�˲���Ƶǿ��ͼ')
    subplot(133),imshow(g4, []),title('-8������˹�˲���Ƶǿ��ͼ')

elseif(n == 10)
    % �����Կռ��˲���
    f = imread('Fig0219(a).tif');
    fn = imnoise(f, 'salt & pepper', 0.2);  % ��ӽ�������
    g1 = medfilt2(fn);              % ��ֵ�˲� 0���
    g2 = medfilt2(fn, 'symmetric'); % ��ֵ�˲� �߽�Գ����

    figure;
    subplot(221),imshow(f),title('ԭͼ');
    subplot(222),imshow(fn),title('����������');
    subplot(223),imshow(g1),title('��ֵ�˲� 0���');
    subplot(224),imshow(g2),title('��ֵ�˲� �߽�Գ����');

end