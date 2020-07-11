function course_ch3(n)
% ������ Ƶ����
    % n=1   ��ά��ɢ����Ҷ�任
    % n=2   Ƶ���˲�
    % n=3   �ռ����˲� => Ƶ�����˲�
    % n=4   �����ͨƵ���˲���
    % n=5   ������˹��ͨƵ���˲���
    % n=6   ��˹��ͨ�˲���
    % n=7   �߿�ͼ�����ͼ
    % n=8   ��ͨ�˲���
    % n=9   ��ͨ�˲���
    % n=10  ��ͨ�˲�
    % n=11  ��Ƶǿ���˲�
    % n=12  �ݲ������˲���cnotch����
    % n=13  �ݲ������˲���recnotch����

if(n == 1)
    % ��ά��ɢ����Ҷ�任
    % ==============================ԭͼ
    f = imread('Fig0303(a).tif');
    % f = im2single(f);
    F = fft2(f);            % DFT
    f_ = ifft2(F);          % IDFT
    f_ = real(f_);          % ��ȥ���
    Fc = fftshift(F);       % �ƶ�Ƶ������
    SF = abs(Fc);           % ԭͼƵ��ͼ
    SF2 = log(1 + abs(Fc)); % �����任Ƶ��ͼ
    fphi = angle(Fc);       % ���
    fphi = atan2(imag(F), real(F));
    [M, N] = size(f);
    % fprintf("Ƶ�����ģ�(%d, %d)\n", floor(M/2)+1, floor(N/2)+1);
    % ==============================ƽ��
    g = zeros(512);
    g(209+50:305+50, 249-80:265-80) = f(209:305, 249:265);
    % g = im2single(g);
    G = fft2(g);            % DFT
    Gc = fftshift(G);       % �ƶ�Ƶ������
    SG = abs(Gc);           % ԭͼƵ��ͼ
    SG2 = log(1 + abs(Gc)); % �����任Ƶ��ͼ
    gphi = angle(Gc);       % ���
    % ==============================��ת
    h = imrotate(f, -45, 'bilinear', 'crop');
    H = fft2(h);            % DFT
    Hc = fftshift(H);       % �ƶ�Ƶ������
    SH = abs(Hc);           % ԭͼƵ��ͼ
    SH2 = log(1 + abs(Hc)); % �����任Ƶ��ͼ
    hphi = angle(Hc);       % ���
    %?==============================��ͼ
    figure;
    subplot(331), imshow(f), title('ԭͼ');
    subplot(332), imshow(SF2, []), title('�����任Ƶ��ͼ');
    subplot(333), imshow(fphi, []), title('ԭͼ���');
    subplot(334), imshow(g), title('ƽ��ԭͼ');
    subplot(335), imshow(SG2, []), title('�����任Ƶ��ͼ');
    subplot(336), imshow(gphi, []), title('ƽ�����');
    subplot(337), imshow(h), title('��תԭͼ');
    subplot(338), imshow(SH2, []), title('�����任Ƶ��ͼ');
    subplot(339), imshow(hphi, []), title('��ת���');

elseif(n == 2)
    % Ƶ���˲�
    f = imread('Fig0305(a).tif');       % ��ȡͼ��
    figure,
    subplot(221),imshow(f),title('ԭͼ');

    % �����Ч��
    f = imread('Fig0305(a).tif');       % ��ȡͼ��
    [M, N] = size(f);                   % ͼ���С
    [f, revertclass] = tofloat(f);      % תΪfloat����
    F = fft2(f);                        % DFT
    H = lpfilter('gaussian', M, N, 10); % ������Ӧ�ĸ���Ҷ�任(���ݺ���) 10��ʾ��׼��
    G = H.*F;                           % Ƶ�����˲�
    g = ifft2(G);                       % DFT���任
    gc = revertclass(g);                % ת��unit8����
    subplot(222),imshow(gc),title('�����Ч��');

    % ���Ч�� ��˷���
    f = imread('Fig0305(a).tif');       % ��ȡͼ��
    [f, revertclass] = tofloat(f);      % תΪfloat����
    PQ = paddedsize(size(f));           % ��������С
    F = fft2(f, PQ(1), PQ(2));          % DFT
    H = lpfilter('gaussian', PQ(1), PQ(2), 2*10);
    G = H.*F;                           % Ƶ�����˲�
    g = ifft2(G);                       % DFT���任
    gc = g(1:size(f,1), 1:size(f,2));   % ��ȡ��С
    gc = revertclass(gc);               % ת��unit8����
    subplot(223),imshow(gc),title('���Ч����Ƶ�����˲���');

    % �������(�����Ч��һ��)
    f = imread('Fig0305(a).tif');       % ��ȡͼ��
    [f, revertclass] = tofloat(f);      % תΪfloat����
    h = fspecial('gaussian', 15, 7);    % ���ɸ�˹ģ��
    g = imfilter(f, h);                 % �˲���
    g = revertclass(g);                 % ת��unit8����
    subplot(224), imshow(g), title('������ռ����˲���');

elseif(n == 3)
    % �ռ����˲� => Ƶ�����˲�
    f = imread('Fig0309(a).tif');
    f = tofloat(f);
    F = fft2(f);                    % DFT
    Fc = fftshift(log(1+abs(F)));   % Ƶ������ת�����Ͻ�
    PQ = paddedsize(size(f));       % ������Ҫ������ֵ
    h = fspecial('sobel')';         % ����ת180��
    H = freqz2(h, PQ(1), PQ(2));    % DFT
    H = ifftshift(H);               % Ƶ�������Ƶ����Ͻ�
    gs = imfilter(f, h);            % �ռ����˲�
    gf = dftfilt(f, H);             % Ƶ�����˲�

    figure;
    subplot(221),imshow(f),title('ԭͼ');
    subplot(222),imshow(Fc, []),title('���������Ƶ��');
    subplot(223),imshow(abs(gs), []),title('�ռ����˲�');
    subplot(224),imshow(abs(gf), []),title('Ƶ�����˲�');

elseif(n == 4)
    % �����ͨƵ���˲���
    f = imread('Fig0313(a).tif');
    [f, revertclass] = tofloat(f);
    PQ = paddedsize(size(f));
    [U, V] = dftuv(PQ(1), PQ(2));
    D = hypot(U, V);
    D0 = 0.05*PQ(2);            % �����ֹƵ��
    F = fft2(f, PQ(1), PQ(2));
    H = single(D <= D0);        % �����˲���
    g = dftfilt(f, H);
    g = revertclass(g);
    
    figure;
    subplot(221),imshow(f),title('ԭͼ');
    subplot(222),imshow(log(1+abs(fftshift(F))), []),title('ԭͼ����Ƶ��');
    subplot(223),imshow(g),title('�˲�');
    subplot(224),imshow(fftshift(H)),title('�����ͨ�˲���');

elseif(n == 5)
    % ������˹��ͨƵ���˲���
    f = imread('Fig0313(a).tif');
    [f, revertclass] = tofloat(f);
    PQ = paddedsize(size(f));
    [U, V] = dftuv(PQ(1), PQ(2));
    D = hypot(U, V);
    D0 = 0.05*PQ(2);            % �����ֹƵ��
    F = fft2(f, PQ(1), PQ(2));
    n = 2;
    H = 1./(1+(D./D0).^(2*n));  % ������˹��ͨ�˲���
    g = dftfilt(f, H);
    g = revertclass(g);
    
    figure;
    subplot(221),imshow(f),title('ԭͼ');
    subplot(222),imshow(log(1+abs(fftshift(F))), []),title('ԭͼ����Ƶ��');
    subplot(223),imshow(g),title('�˲�');
    subplot(224),imshow(fftshift(H)),title('������˹��ͨ�˲���');

elseif(n == 6)
    % ��˹��ͨ�˲���
    f = imread('Fig0313(a).tif');
    [f, revertclass] = tofloat(f);
    PQ = paddedsize(size(f));
    [U, V] = dftuv(PQ(1), PQ(2));
    D = hypot(U, V);
    D0 = 0.05*PQ(2);            % �����ֹƵ��
    F = fft2(f, PQ(1), PQ(2));
    H = exp(-D.^2/(2*(D0^2)));  % ��˹��ͨ�˲���
    g = dftfilt(f, H);
    g = revertclass(g);
    
    figure;
    subplot(221),imshow(f),title('ԭͼ');
    subplot(222),imshow(log(1+abs(fftshift(F))), []),title('ԭͼ����Ƶ��');
    subplot(223),imshow(g),title('�˲�');
    subplot(224),imshow(fftshift(H)),title('��˹��ͨ�˲���');

elseif(n == 7)
    % �߿�ͼ�����ͼ
    H = fftshift(lpfilter('gaussian', 500, 500, 50));
    figure;
    mesh(double(H(1:10:500, 1:10:500)));        % �߿�ͼ
    axis tight, axis off, colormap([0 0 0]);
    figure;
    surf(double(H(1:10:500, 1:10:500)));        % ����ͼ
    axis tight, axis off, colormap(gray);
    shading interp;

elseif(n == 8)
    %  ��ͨ�˲���
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
    subplot(221),imshow(f),title('ԭͼ');
    subplot(222),imshow(g1),title('ideal�˲�');
    subplot(223),imshow(g2),title('btw�˲�');
    subplot(224),imshow(g3),title('gaussian�˲�');

elseif(n == 9)
    % ��ͨ�˲���
    H1 = fftshift(hpfilter('ideal', 500, 500, 50));      % �����ͨ�˲���
    H2 = fftshift(hpfilter('btw', 500, 500, 50, 2));    % ������˹��ͨ�˲���
    H3 = fftshift(hpfilter('gaussian', 500, 500, 50));  % ��˹��ͨ�˲���
    figure;
    subplot(231),imshow(H1, []), title('�����ͨ�˲��� 2D');
    subplot(234),mesh(double(H1(1:10:end, 1:10:end))),title('ideal D0=50'),axis off;
    subplot(232),imshow(H2, []), title('������˹��ͨ�˲��� 2D');
    subplot(235),mesh(double(H2(1:10:end, 1:10:end))),title('btw D0=50 n=2'),axis off;
    subplot(233),imshow(H3, []), title('��˹��ͨ�˲��� 2D');
    subplot(236),mesh(double(H3(1:10:end, 1:10:end))),title('gaussian D0=50'),axis off;
    
elseif(n == 10)
    % ��ͨ�˲�
    f = imread('Fig0313(a).tif');
    PQ = paddedsize(size(f));
    f = tofloat(f);
    D0 = 0.05 * PQ(1);
    H = hpfilter('gaussian', PQ(1), PQ(2), D0);
    g = dftfilt(f, H);
    figure;
    subplot(121),imshow(f),title('ԭͼ');
    subplot(122),imshow(g),title('�˲���');

elseif(n == 11)
    % ��Ƶǿ���˲�
    f = imread('Fig0319(a).tif');
    PQ = paddedsize(size(f));
    D0 = 0.05 * PQ(1);
    HBW = hpfilter('btw', PQ(1), PQ(2), D0, 2); % btw�˲���
    H = 0.5 + 2*HBW;                            % ��Ƶǿ���˲���
    g1 = dftfilt(f, HBW, 'fltpoint');
    g1 = gscale(g1);                % ����ͼ��Ԫ�ش�С��[0, 255]
    g2 = dftfilt(f, H, 'fltpoint'); % float����
    g2 = gscale(g2);                % ����ͼ��Ԫ�ش�С��[0, 255]
    g2_eq = histeq(g2, 256);        % ��g2����ֱ��ͼ����
    g3 = dftfilt(f, H);             % uint8����
    g3_eq = histeq(g3, 256);        % ��g3����ֱ��ͼ����

    figure;
    subplot(221),imshow(f),title('ԭͼ');
    subplot(222),imhist(f);title('ԭͼֱ��ͼ');
    subplot(223),imshow(g1),title('btw�˲�');
    subplot(224),imhist(g1);title('btw�˲�ֱ��ͼ');
    figure;
    subplot(221),imshow(g2),title('g2 ��Ƶ�˲�');
    subplot(222),imhist(g2);title('g2 ��Ƶ�˲�ֱ��ͼ');
    subplot(223),imshow(g2_eq),title('g2 ��Ƶ�˲� ֱ��ͼ�����');
    subplot(224),imhist(g2_eq);title('g2 ��Ƶ�˲� ֱ��ͼ����');
    figure;
    subplot(221),imshow(g3),title('g3 ��Ƶ�˲�');
    subplot(222),imhist(g3);title('g3 ��Ƶ�˲�ֱ��ͼ');
    subplot(223),imshow(g3_eq),title('g3 ��Ƶ�˲� ֱ��ͼ�����');
    subplot(224),imhist(g3_eq);title('g3 ��Ƶ�˲� ֱ��ͼ����');

elseif(n == 12)
    % �ݲ������˲���cnotch����
    f = imread('Fig0321(a).tif');
    [f, revertclass] = tofloat(f);
    [M, N] = size(f);
    F = fft2(f);                        % DFT
    Fc = log(1 + abs(fftshift(F)));                     % ����Ƶ��
    S = gscale(Fc);     % imtool(S)������ü������
    C1 = [99 154; 128 163];
    H1 = cnotch('gaussian', 'reject', M, N, C1, 5);     % �ݲ������˲���
    p1 = gscale(fftshift(H1).*(tofloat(S)));
    g1 = dftfilt(f, H1);                                % �ݲ�����C1 Ƶ���˲�
    g1 = revertclass(g1);
    figure;
    subplot(121),imshow(f),title('ԭͼ');
    subplot(122),imshow(S),title('ԭͼ����Ƶ��');
    figure;
    subplot(121),imshow(p1),title('�ݲ������˲� ');
    subplot(122),imshow(g1),title('�ݲ������˲� ');

    C2 = [99 154; 128 163; 49 160; 133 233; 55 132; 108 225; 112 74];
    H2 = cnotch('gaussian', 'reject', M, N, C2, 5);     % �ݲ������˲���
    p2 = gscale(fftshift(H2).*(tofloat(S)));
    g2 = dftfilt(f, H2);                                % �ݲ�����C2 Ƶ���˲�
    g2 = revertclass(g2);
    figure;
    subplot(121),imshow(p2),title('�ݲ������˲� ');
    subplot(122),imshow(g2),title('�ݲ������˲� ');

elseif(n == 13)
    %�ݲ������˲���recnotch����
    f = imread('Fig0322(a).tif');
    [f, revertclass] = tofloat(f);
    [M, N] = size(f);
    F = fft2(f);
    Fc = log(1 + abs(fftshift(F)));
    S = gscale(Fc);
    HR = recnotch('reject', 'vertical', M, N, 3, 15, 15);    % �ݲ������˲���
    g1 = dftfilt(f, HR);            % Ƶ���˲� ȥ���ռ���� ��ȥ��
    g1 = revertclass(g1);
    HP = recnotch('pass', 'vertical', M, N, 3, 15, 15);      % �ݲ���ͨ�˲���
    g2 = dftfilt(f, HP);            % Ƶ���˲� �õ��ռ���� ������
    g2 = gscale(g2);

    figure;
    subplot(121),imshow(f),title('ԭͼ');
    subplot(122),imshow(S),title('ԭͼ����Ƶ��');
    figure;
    subplot(121),imshow(HR),title('�ݲ������˲���');
    subplot(122),imshow(g1),title('�ݲ������˲� ȥ��');
    figure;
    subplot(121),imshow(HP),title('�ݲ���ͨ�˲���');
    subplot(122),imshow(g2),title('�ݲ���ͨ�˲� ����');

end