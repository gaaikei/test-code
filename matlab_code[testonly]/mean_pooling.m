function [ im_1 ] = mean_pooling( im )
%MEAN_POOLING Summary of this function goes here
%   Detailed explanation goes here
s = size(im);
im_1 = zeros(s(1),s(2),3);
l = s(3)/3;
for i=1:l
	im_1(:,:,1) = im(:,:,i)+im(:,:,1) ;
	im_1(:,:,2) = im(:,:,i)+im(:,:,2) ;
	im_1(:,:,3) = im(:,:,i)+im(:,:,3) ;
end
im_1(:,:,1)=im_1(:,:,1)/l;
im_1(:,:,2)=im_1(:,:,2)/l;
im_1(:,:,3)=im_1(:,:,3)/l;
end

