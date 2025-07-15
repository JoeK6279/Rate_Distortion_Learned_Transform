function y = idct_viii(x)

N = length(x);
M = sqrt(4/(2*N+1))*cos(pi*(2*(0:N-1)'+1)*(2*(0:N-1)+1)/(4*N+2));
y = M*x;