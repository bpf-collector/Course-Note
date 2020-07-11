function H = add2(h1, h2)
    % 两个带通滤波器相加, 返回带通滤波器
        %   h1, h2为带通滤波器
        %   TYPE: 'reject'(带阻)、'pass'(带通)
        H = h1 + h2;
        H = H - min(H(:));
        H = H / max(H(:));
    
    end