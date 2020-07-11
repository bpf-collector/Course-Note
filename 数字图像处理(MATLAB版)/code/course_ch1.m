function course_ch1(n)
% 第一章 序言 —— 代码优化
    % n=1   代码优化 sinfun_(M)
    % n=2   twodsin_(A, u0, v0, M, N)
    
if(n == 1)
    disp('运行中...');
    M = 0:2500:20000;
    for k = 1:numel(M)
        f1 = @() sinfun1(M); % 函数句柄
        f2 = @() sinfun2(M);
        f3 = @() sinfun3(M);
        t1(k) = timeit(f1);  % 计算函数时间
        t2(k) = timeit(f2);
        t3(k) = timeit(f3);
    end
    figure,
    plot(t1, 'r-'), title('代码优化');
    hold on, plot(t2, 'g--');
    hold on, plot(t3, 'b.-');
elseif(n == 2)
    f = twodsin1(1, 1/(4*pi), 1/(4*pi), 512, 512);
    figure,imshow(f),title('1*sin(1/4*pi*x+1/4*pi*y)');
else
    disp('parameter x input error.');
end
end

function y = sinfun1(M)
    x = 0:M-1;
    for k = 1:numel(x)
        y(k) = sin(x(k) / (100*pi));
    end
end

function y = sinfun2(M)
    x = 0:M-1;
    y = zeros(1, numel(x));
    for k = 1:numel(x)
        y(k) = sin(x(k) / (100*pi));
    end
end

function y = sinfun3(M)
    x = 0:M-1;
    y = sin(x ./ (10*pi));
end

% 代码优化
function f = twodsin1(A, u0, v0, M, N)
    f = zeros(M, N);
    for c = 1:N
        v0y = v0*(c-1);
        for r = 1:M
            u0x = u0 * (r-1);
            f(r, c) = A*sin(u0x + v0y);
        end
    end
end

function f = twodsin2(A, u0, v0, M, N)
    % f(x, y) = A*sin(u*x+v*y)
    % Test: 
    % f = twodsin2(1, 1/(4*pi), 1/(4*pi), 512, 512);
    % imshow(f)
    r = 0:M-1;
    c = 0:N-1;
    [C, R] = meshgrid(c, r);
    f = A*cos(u0*R + v0*C);
end