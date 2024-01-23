function [anchormatrix] = constructanchor(m,anchorrange)
%CONSTRUCTANCHOR 此处显示有关此函数的摘要
%   此处显示详细说明
range = length(anchorrange);

newindex = repmat(anchorrange,1,m);
F = nchoosek(newindex,m);
anchormatrix = unique(F,'rows');

end

