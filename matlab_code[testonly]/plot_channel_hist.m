image = 'test.jpg';
dyn = imread(image);
x = dyn(:,:,1)+100;y = dyn(:,:,2)+100;z = dyn(:,:,3)+100;
x = double(x);y = double(y);z=double(z);
subplot(3,1,1),hist(x,100)
subplot(3,1,2),hist(y,100)
subplot(3,1,3),hist(z,100)