function H = add2(h1, h2)
    % ������ͨ�˲������, ���ش�ͨ�˲���
        %   h1, h2Ϊ��ͨ�˲���
        %   TYPE: 'reject'(����)��'pass'(��ͨ)
        H = h1 + h2;
        H = H - min(H(:));
        H = H / max(H(:));
    
    end