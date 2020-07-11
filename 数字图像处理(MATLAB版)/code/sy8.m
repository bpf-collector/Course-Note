function sy8(n)
% ʵ��� ͼ��ԭ
    % n=1   ȥ�룺�������
    % n=2   ȥ�룺��������
    % n=3   ȥ���

if(n == 1)
    % ȥ�룺�������
    f = imread('./pic/Lena_gray_512.tif');
    f = tofloat(f);
    [M, N] = size(f);
    n = imnoise2('erlang', M, N);           % ����������
    g = f + 0.03*n;
    % figure;
    % subplot(121),imshow(f),title('ԭͼ');
    % subplot(122),imshow(g),title('����ͼ');

    data = load('sy8_1.mat');
    B = data.B;
    c = data.c;
    r = data.r;
    % [B, c, r] = roipoly(g);                 % ѡ��roi����
    [h, npix] = histroi(g, c, r);           % ����roiֱ��ͼ
    [v, unv] = statmoments(h, 2);           % ����mu, sigma
    figure;
    subplot(121),imshow(1-B),title('roi����');
    subplot(122);bar(h, 1),title('roiֱ��ͼ');

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
    % ȥ�룺��������
    f = imread('./pic/Lena_gray_512.tif');
    [f, revertclass] = tofloat(f);
    [M, N] = size(f);

    C = [0 32; 0 64; 16 16; 32 0; 64 0; -16 16];
    A = [0.1  0.3  0.9  0.5  0.01  0.2];
    n = imnoise3(M, N, C, A);           % ������������
    g = n + f;                          % ����ͼ
    G = fft2(g);
    Gc = fftshift(log(1+abs(G)));
    figure;
    subplot(121),imshow(g),title('����ͼ');
    subplot(122),imshow(Gc, []),title('����ͼ����Ƶ��');

    % �����˲�
    D0 = [32, 64, 22.6];
    H = bandfilter2('gaussian', 'reject', M, N, D0, 2, 0);
    g1 = dftfilt(g, H);
    G1 = fft2(g1);
    G1c = fftshift(log(1+abs(G1)));
    figure;
    subplot(121),imshow(g1),title('bandfilter2�˲�');
    subplot(122),imshow(G1c, []),title('bandfilter2�˲�����Ƶ��');

    % �ݲ������˲�
    C1 = [256 288; 256 320; 272 272; 288 256; 320 256; 240 272];
    H = cnotch('gaussian', 'reject', M, N, C1, 2);
    g2 = dftfilt(g, H);
    g2 = revertclass(g2);
    G2 = fft2(g2);
    G2c = fftshift(log(1+abs(G2)));
    figure;
    subplot(121),imshow(g2),title('cnotch�˲�');
    subplot(122),imshow(G2c, []),title('cnotch�˲�����Ƶ��');
elseif(n == 3)
    % ȥ���
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
    % figure;imshow(pixeldup(g1, 6),[]),title('f1���');
    g1 = g1 + n1;            % g1 = h1**f1 + n1;
    % figure;imshow(pixeldup(g1, 6),[]),title('f1�������');
    N1 = fft2(n1);
    % figure;imshow(pixeldup(fftshift(log(1+abs(N1))), 6),[]),title('f1������Ƶ��');

    n2 = imnoise2('gaussian',size(f2,1),size(f2,2),0,sqrt(0.001));
    % figure;imshow(pixeldup(g2, 2),[]),title('f2���');
    g2 = tofloat(g2) + n2;   % g2 = h1**f2 + n2;
    % figure;imshow(pixeldup(g2, 2),[]),title('f2�������');
    N2 = fft2(n2);
    % figure;imshow(pixeldup(fftshift(log(1+abs(N2))), 2),[]),title('f2������Ƶ��');

    R1 = computeR(f1, n1, "f1 ");
    R2 = computeR(f2, n2, "f2 ");
    f1_1 = deconvwnr(g1, h1, 0);       % ֱ�����˲�
    f1_2 = deconvwnr(g1, h1, R1);      % ά���˲�
    figure;
    subplot(121),imshow(pixeldup(f1_1, 4),[]),title("f1 ֱ�����˲�");
    subplot(122),imshow(pixeldup(f1_2, 4),[]),title("f1 Rά���˲�");
    f1_1 = deconvwnr(g1, h1, 0.1);       % ֱ�����˲�
    f1_2 = deconvwnr(g1, h1, 0.5);      % ά���˲�
    figure;
    subplot(121),imshow(pixeldup(f1_1, 4),[]),title("f1 0.1ά���˲�");
    subplot(122),imshow(pixeldup(f1_2, 4),[]),title("f1 0.5ά���˲�");
    f1_1 = deconvwnr(g1, h1, 1);       % ֱ�����˲�
    f1_2 = deconvwnr(g1, h1, 1.5);      % ά���˲�
    figure;
    subplot(121),imshow(pixeldup(f1_1, 4),[]),title("f1 1.0ά���˲�");
    subplot(122),imshow(pixeldup(f1_2, 4),[]),title("f1 1.5ά���˲�");

    f2_1 = deconvwnr(g2, h2, 0);       % ֱ�����˲�
    f2_2 = deconvwnr(g2, h2, R2);      % ά���˲�
    figure;
    subplot(121),imshow(pixeldup(f2_1, 4),[]),title("f2 ֱ�����˲�");
    subplot(122),imshow(pixeldup(f2_2, 4),[]),title("f2 Rά���˲�");
end
end

function R = computeR(f, noise, plot)
    % �������Թ��ʱ�
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