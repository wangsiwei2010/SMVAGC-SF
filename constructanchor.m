function [anchormatrix] = constructanchor(m,anchorrange)
%CONSTRUCTANCHOR �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
range = length(anchorrange);

newindex = repmat(anchorrange,1,m);
F = nchoosek(newindex,m);
anchormatrix = unique(F,'rows');

end

