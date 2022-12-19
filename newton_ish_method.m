function [ epsilon_2 ] = newton_ish_method( f_x_minus_epsilon , f_x , f_x_plus_epsilon )
%  NEWTON_ISH_METHOD guess the x position of the minimum value from
%   f(x-ε), f(x), and f(x+ε).
%   It is roughly based in the Newton minimization method.
%  Angel Rodes, 2022

x1=-1;
x2=0;
x3=1;
y1=f_x_minus_epsilon;
y2=f_x;
y3=f_x_plus_epsilon;

denom = (x1 - x2).*(x1 - x3).*(x2 - x3);
A = (x3 .* (y2 - y1) + x2 .* (y1 - y3) + x1 .* (y3 - y2)) ./ denom;
B = (x3.^2 .* (y1 - y2) + x2.^2 .* (y3 - y1) + x1.^2 .* (y2 - y3)) ./ denom;
% C = (x2 .* x3 .* (x2 - x3) .* y1 + x3 .* x1 .* (x3 - x1) .* y2 + x1 .* x2 .* (x1 - x2) .* y3) ./ denom;

xv=-B ./ (2*A); % x position of the vertex of the parabola that goes through (x1,y1), (x2,y2), and (x3,y3)

epsilon_2=(sign(A)>0)*xv+...
   (sign(A)<0)*(f_x_plus_epsilon>f_x_minus_epsilon)*(f_x_plus_epsilon+1-xv)+...
   (sign(A)<0)*(f_x_plus_epsilon<f_x_minus_epsilon)*(f_x_minus_epsilon-1-xv);

end

