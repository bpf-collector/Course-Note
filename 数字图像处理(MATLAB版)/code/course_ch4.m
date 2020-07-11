function course_ch4(n)
% ������ ͼ��ԭ
    % n=1   imnoise �������
    % n=2   imnoise2 �������
    % n=3   imnoise3 ��������
    % n=4   ������������
    % n=5   ���������ĸ�ԭ ȥ��
    % n=6   �˻�����ģ��
    % n=7   ���о���ĸ�ԭ ȥ���

    if(n == 1)
        % imnoise �������
        f = imread("./pic/Lena.jpg");
        f = rgb2gray(f);
        f = tofloat(f);
        g1 = imnoise(f, 'gaussian', 0, 0.01);   % [��ֵ, ����]
        g2 = imnoise(f, 'poisson');

        fig_plot(f, g1, "gaussian");
        fig_plot(f, g2, "poisson");

    elseif(n == 2)
        % imnoise2 �������
        f = imread("./pic/Lena.jpg");
        f = rgb2gray(f);
        f = tofloat(f);
        [M, N] = size(f);
        noise1 = imnoise2('gaussian', M, N, 0, 0.1); % [��ֵ, ��׼��]
        g1 = noise1 + f;
        noise2 = imnoise2('uniform', M, N);
        g2 = noise2 + f;
        noise3 = imnoise2('rayleigh', M, N);
        g3 = 0.25*noise3 + f;

        fig_plot(f, g1, "gaussian");
        fig_plot(f, g2, "uniform");
        fig_plot(f, g3, "rayleigh");
    
    elseif(n == 3)
        % imnoise3 ��������
        f = imread("./pic/Lena.jpg");
        f = rgb2gray(f);
        f = tofloat(f);
        [M, N] = size(f);
        C = [2 4; 30 -6];
        [n, N, S] = imnoise3(M, N, C);
        g = n + f;
        figure;imshow(S, []),title("����ͼ����Ƶ��")
        [x, y] = find(S>0.5);
        fig_plot(f, g, "imnois3");

    elseif(n == 4)
        % ������������
        f = imread('Fig0404(a).tif');
        [B, c, r] = roipoly(f);         % ���roi����
        [p, npix] = histroi(f, c, r);   % ����roiֱ��ͼ
        [v, unv] = statmoments(p, 2);   % �������ľ�
        % u = unv(1); sigma = unv(2);
        X = imnoise2('gaussian', npix, 1, 147, 20);
        figure;
        subplot(121),bar(p,1);title('����roi����ֱ��ͼ');
        subplot(122),hist(X, 130);title('gaussian����');
    elseif(n == 5)
        % ���������ĸ�ԭ(�ռ��˲�) g = f + n  �ռ������˲�
        f = imread('Fig0219(a).tif');
        [M, N] = size(f);

        % ����
        R1 = imnoise2('salt & pepper', M, N, 0.1, 0);
        g1 = f;
        g1(R1==0) = 0;   % ��������������Ⱦ��ͼ�� �ڵ�
        R2 = imnoise2('salt & pepper', M, N, 0, 0.1);
        g2 = f;
        g2(R2==0) = 255; % ��������������Ⱦ��ͼ�� �׵�
        figure;
        subplot(131),imshow(f),title('ԭͼ');
        subplot(132),imshow(g1,[]),title('��������������Ⱦ');
        subplot(133),imshow(g2,[]),title('��������������Ⱦ');

        % �����;�ֵ�˲�ȥ��
        f1 = spfilt(g1, 'chmean', 3, 3, 1.5);   % ȥ���ڵ�Q>0
        f2 = spfilt(g2, 'chmean', 3, 3, -1.5);  % ȥ���׵�Q<0
        figure;
        subplot(121),imshow(g1,[]),title('�����;�ֵȥ����������');
        subplot(122),imshow(g2,[]),title('�����;�ֵȥ����������');

        % �����Сֵ�˲�ȥ��
        f1 = spfilt(g1, 'max', 3, 3);
        f2 = spfilt(g2, 'min', 3, 3);
        figure;
        subplot(121),imshow(g1,[]),title('���ֵȥ����������');
        subplot(122),imshow(g2,[]),title('��Сֵȥ����������');

        % ����Ӧ�ռ��˲���
        %   ����
        g = imnoise(f, 'salt & pepper', 0.25);
        f1 = medfilt2(g, [7 7], 'symmetric');
        f2 = adpmedian(g, 7);
        figure;
        subplot(121),imshow(f1),title('��ֵȥ����������');
        subplot(122),imshow(f2),title('����Ӧ��ֵȥ����������');

    elseif(n == 6)
        % �˻�����ģ�͡���g = H(f) + n = h ** f + n;
        f = checkerboard(8);
        [M, N] = size(f);
        h = fspecial('motion', 7, 45);
        gb = imfilter(f, h, 'circular');
        n = imnoise2('gaussian', M, N, 0, sqrt(0.001));
        g = gb + n;
        figure;
        subplot(221),imshow(pixeldup(f, 8), []),title('ԭͼ');
        subplot(222),imshow(pixeldup(gb, 8), []),title('ԭͼ���');
        subplot(223),imshow(pixeldup(g, 8), []),title('����ͼ');
        subplot(224),imshow(pixeldup(n, 8), []),title('����');

    elseif(n == 7)
        % ֱ�����˲� ά���˲�
        f = checkerboard(8);
        [M, N] = size(f);
        h = fspecial('motion', 7, 45);  % �˲���

        n = imnoise2('gaussian', M, N, 0, sqrt(0.001)); % ����
        Sn = abs(fft2(n)).^2;
        nA = sum(Sn(:)) / (M*N);
        Sf = abs(fft2(f)).^2;
        fA = sum(Sf(:)) / (M*N);
        R = nA / fA;                    % ���Ź��ʱ�

        gb = imfilter(f, h, 'circular');% ���
        g = gb + n;                     % ����ͼ
        f1 = deconvwnr(g, h);           % ֱ�����˲�
        f2 = deconvwnr(g, h, R);        % ά���˲�
        NCORR = fftshift(real(ifft2(Sn)));
        ICORR = fftshift(real(ifft2(Sf)));
        f3 = deconvwnr(g, h, NCORR, ICORR); % ��������˲�
        figure;
        subplot(221),imshow(pixeldup(g, 8)),title('����ͼ');
        subplot(222),imshow(pixeldup(f1, 8)),title('ֱ�����˲�');
        subplot(223),imshow(pixeldup(f2, 8)),title('���Ź��ʱ����˲�');
        subplot(224),imshow(pixeldup(f3, 8)),title('��������˲�');

    end
end

function fig_plot(f, g, type)
    n = g - f;
    figure;
    subplot(221),imshow(f),title('ԭͼ');
    subplot(222),imshow(g),title('����');
    subplot(223),imshow(n, []),title(type+"����");
    subplot(224),hist(n, 50),title('����ֱ��ͼ');
end