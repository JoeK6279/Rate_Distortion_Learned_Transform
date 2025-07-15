function y = dst_vii(x)

N = length(x);
M = sqrt(4/(2*N+1))*sin(pi*(2*(0:N-1)'+1)*((0:N-1)+1)/(2*N+1));
y = M'*x;