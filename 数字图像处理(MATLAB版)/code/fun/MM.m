function [Max, Min] = MM(A)
% 计算最大最小值
    Max = max(A(:));
    Min = min(A(:));
    fprintf("max=%f, min=%f\n",Max,Min);