function course_ch5(n)
% ������ ��ɫͼ����
    % n=1   RGBͼ��
    % n=2   ����ͼ��
    % n=3   ����ͼ��
    % n=4   Ԥ����Ĳ�ɫӳ��
    % n=5   ����RGB������ͼ��
    % n=6   ��ɫͼ��ƽ������
    % n=7   ��ɫͼ���񻯴���
    % n=8   �ݶȼ���ɫ��Ե
    % n=9   ͼ��ָ�

    if(n == 1)
        % RGBͼ��
        rgbcube();
    elseif(n == 2)
        % ����ͼ��
        load mandrill;
        figure;imshow(X, map),title('����ͼ��');
    elseif(n == 3)
        % ����ͼ��
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
        % Ԥ����Ĳ�ɫӳ��
        load mandrill;
        colormap(copper);
        subplot(121), imshow(X, map),title('����ͼ��');
        subplot(122), imshow(X, copper),title('Ԥ����Ĳ�ɫӳ��');
    elseif(n == 5)
        % ����RGB������ͼ��
        f = imread('Fig0604(a).tif');
        [X1, map1] = rgb2ind(f, 8, 'nodither');
        %  8 ��ʾ�����ͼ��ֻ��8����ɫ
        [X2, map2] = rgb2ind(f, 8, 'dither');
        figure;
        subplot(131),imshow(f),title('ԭͼ');
        subplot(132),imshow(X1, map1),title('�Ƕ�������');
        subplot(133),imshow(X2, map2),title('��������');
        g = rgb2gray(f);
        g1 = dither(g);
        figure;
        subplot(121),imshow(g),title('�Ҷ�ͼ��');
        subplot(122),imshow(g1),title('��������');
    elseif(n == 6)
        % ��ɫͼ��ƽ������
        f = imread('Fig0622(a).tif');
        [f, revertclass] = tofloat(f);
        w = fspecial('average', 25);
        % ======================================= rgbͼ��ƽ������
        f1 = imfilter(f, w, 'replicate'); % ������ͬ�ڶԷ����ֱ��˲�
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
        subplot(121),imshow(f),title('ԭͼ');
        subplot(122),imshow(f1),title('ƽ������');
        % ======================================= hsiͼ��ƽ������
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
        subplot(131),imshow(f),title('ԭͼ');
        subplot(132),imshow(f1),title('�˲�I����');
        subplot(133),imshow(f2),title('�˲�HSI����');
    elseif(n == 7)
        % ��ɫͼ���񻯴���
        f = imread('Fig0622(a).tif');
        [f, revertclass] = tofloat(f);
        w = fspecial('average', 5);
        f1 = imfilter(f, w, 'replicate');
        lapmask = [1 1 1; 1 -8 1; 1 1 1];
        f2 = f1 - imfilter(f1, lapmask, 'replicate');
        figure;
        subplot(131),imshow(f),title('ԭͼ');
        subplot(132),imshow(f1),title('�˲�ͼ');
        subplot(133),imshow(f2),title('��ͼ');
    elseif(n == 8)
        % �ݶȼ���ɫ��Ե
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
        subplot(131),imshow(VG),title('�����ݶ�');
        subplot(132),imshow(PRG),title('�����ݶ���͵�ͼ��');
        subplot(133),imshow(err,[]),title('VG-PRG');
    elseif(n == 9)
        % ͼ��ָ�
        f = imread('Fig0630(a).tif');
        mask = roipoly(f);          % ѡȡroi����
        fr = immultiply(mask, f(:,:,1));
        fg = immultiply(mask, f(:,:,2));
        fb = immultiply(mask, f(:,:,3));
        g = cat(3, fr, fg, fb);
        [M, N, K] = size(g);
        I = reshape(g, M*N, K);     % K=3
        idx = find(mask);
        I = double(I(idx, :));
        [C, m] = covmatrix(I);      % ����Э�������C�;�ֵm
        d = diag(C);                % ����dΪλ��Э�������C�ĶԽ�����
        sd = sqrt(d);               % ��׼��sd
        T = max(ceil(sd));          % ��ֵΪ��׼�����ֵ������ȡ��
        figure;
        subplot(121),imshow(f),title('ԭͼ');
        subplot(122),imshow(g),title('roi����');
        % ================================= ŷʽ�����㷨
        T12 = colorseg('euclidean', f, 1*T, m);
        T24 = colorseg('euclidean', f, 2*T, m);
        T48 = colorseg('euclidean', f, 4*T, m);
        T96 = colorseg('euclidean', f, 8*T, m);
        figure;
        subplot(221),imshow(T12),title('ŷʽ T=12');
        subplot(222),imshow(T24),title('ŷʽ T=24');
        subplot(223),imshow(T48),title('ŷʽ T=48');
        subplot(224),imshow(T96),title('ŷʽ T=96');
        % ================================= ��ʽ�����㷨
        T12 = colorseg('mahalanobis', f, 1*T, m, C);
        T24 = colorseg('mahalanobis', f, 2*T, m, C);
        T48 = colorseg('mahalanobis', f, 4*T, m, C);
        T96 = colorseg('mahalanobis', f, 8*T, m, C);
        figure;
        subplot(221),imshow(T12),title('��ʽ T=12');
        subplot(222),imshow(T24),title('��ʽ T=24');
        subplot(223),imshow(T48),title('��ʽ T=48');
        subplot(224),imshow(T96),title('��ʽ T=96');
    end
end