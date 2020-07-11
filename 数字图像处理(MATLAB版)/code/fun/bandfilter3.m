function H = bandfilter3(type, M, N, Radii, Width, Order)
%������˹���衢��ͨ�˲���
%   TYPE: 'reject'(����)��'pass'(��ͨ)
%   M, N: the number of rows and columns in the filter
%   Radii: ��ʾÿ��Ƶ�����ĵ�D0��1*k����
%   Width: ��ʾÿ��Ƶ���Ŀ�ȣ�1*k����
%   Order: ��ʾÿ��Ƶ���Ľ�����1*k����

    [U, V] = dftuv(M, N);
    D = hypot(U, V);
    k = numel(Radii);       % Ƶ������

    if(k == 1)          % һ��Ƶ��
        H = btwReject(D, Radii, Width, Order);  % �����˲���

    elseif(k == 2)      % ����Ƶ��
        if(numel(Order) == 1)
            Order(2) = Order;       % ��Ϊ1*k����
        end
        h1 = btwReject(D, Radii(1), Width(1), Order(1));  % �����˲���h1
        h2 = btwReject(D, Radii(2), Width(2), Order(2));  % �����˲���h2
        H = add_bandfilter(h1, h2);                       % �˲������
    end

    if strcmp(type, 'pass')
        % ��ͨ�˲���
        H = 1 - H;
    end
end

function H = add_bandfilter(h1, h2)
% ������ͨ�˲������, ���ش�ͨ�˲���
    %   h1, h2Ϊ��ͨ�˲���
    H = h1 + h2;
    H = H - min(H(:));
    H = H / max(H(:));
end

function H = btwReject(D, D0, W, n)
% �����˲���
    H = 1 ./ (1 + (((D*W)./(D.^2-D0^2 + eps)).^(2*n)));
end
