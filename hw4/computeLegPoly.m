function [ z ] = computeLegPoly( x, Q )
%COMPUTELEGPOLY Return the Qth order Legendre polynomial of x
%   Inputs:
%       x: vector (or scalar) of reals in [-1, 1]
%       Q: order of the Legendre polynomial to compute
%   Output:
%       z: matrix where each column is the Legendre polynomials of order 0 
%          to Q, evaluated at the corresponding x value in the input
z=[];
for index1=1:length(x)
    zCol = [];
    for index2=1:Q
        y = legendreP(index2,x(index1));
        zCol = [zCol, y];
    end
    zCol = [1,zCol];
    z = [z; zCol];
end

end