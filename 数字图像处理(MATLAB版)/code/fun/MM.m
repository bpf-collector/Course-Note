function [Max, Min] = MM(A)
% ���������Сֵ
    Max = max(A(:));
    Min = min(A(:));
    fprintf("max=%f, min=%f\n",Max,Min);