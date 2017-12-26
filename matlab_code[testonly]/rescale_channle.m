function [ im ] = rescale_channle( im )
%RESCALE_CHANNLE Summary of this function goes here
%   Detailed explanation goes here
s = size(im);
if s(3) ~= 3
	printf ('worng channel numbers');
	exit();
end
for i=1:3
	max_n = max(max(im(:,:,i)));
	min_n = min(min(im(:,:,i)));
	im(:,:,i) = im(:,:,i)-min_n;
	im(:,:,i) = im(:,:,i)/max_n;
	im(:,:,i) = im(:,:,i)*255;
end
%im = uint8(im);

