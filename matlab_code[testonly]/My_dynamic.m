images = cell(1,30);
l=length(images);
grayimage = cell(1,30);

dirpath = './test_data/';
path = dir(dirpath);
temp = 1;
for i=1:l
	if strcmp(path(i).name,'.') || strcmp(path(i).name,'..')
		path(i).name=[];
		temp = temp + 1;
		continue;
	end
	imagename = path(i).name;
	imagepath = strcat(dirpath,imagename);
	im = imread(imagepath);
	images{1,i- temp + 1}= im;
	im = rgb2gray(im);
	grayimage{1,i-temp+1} = im;
end
% imshow(images{1,3});
% for i=1:29
% 	imshow(uint8(grayimage{1,i}-grayimage{1,i+1}));
% end
% 
% for i=1:29
% 	im(grayimage{1,i}-grayimage{1,i+1});
% end


im_1 = zeros(240,320,16);

for i=1:16
	im_1(:,:,i) = (grayimage{1,i+1}-grayimage{1,i});
% 	im_1(:,:,i) = (images{1,i+1}-images{1,i});
end
im = zeros(240,320,3);
im(:,:,1) = (im_1(:,:,1)+im_1(:,:,2)+im_1(:,:,3)+im_1(:,:,4)+im_1(:,:,5))/5;
im(:,:,2) = (im_1(:,:,6)+im_1(:,:,8)+im_1(:,:,9)+im_1(:,:,10)+im_1(:,:,7))/5;
im(:,:,3) = (im_1(:,:,11)+im_1(:,:,12)+im_1(:,:,13)+im_1(:,:,14)+im_1(:,:,15))/5;
%imshow(uint8(im));

% im = mean_pooling(im_1);


%rescale every channel
% im = im(:,:,1) + 20;
 im = rescale_channle(im);

imshow(uint8(im));



% im = im - min(im(:)) ;
% im = 255 * (im ./ max(im(:)));
% imshow(uint8(im));