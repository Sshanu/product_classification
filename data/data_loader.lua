require 'image'
img_size = 48
l = 17
ibt=torch.ByteTensor(l*4*(8+3*9),3,img_size,img_size)
il=torch.ByteTensor(l*4*(8+3*9))
j = 1
s = {'croped','diff','difflight','light'}
for x=1,4 do

	for i=1,l do
	    local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	    local img = image.scale(img,img_size,img_size)
	    ibt[j] = img
			il[j] = i
	    j=j+1; print(j)
	end
	for i=1,l do
	    local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	    local img = image.scale(img,500,500):double()
	    local img = image.crop(img, 0, 0, 490, 490)
	    local img = image.scale(img,img_size,img_size)
	    ibt[j] = img
	    il[j] = i
	    j=j+1; print(j)
	end
	for i=1,l do
	    local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	    local img = image.scale(img,500,500):double()
	    local img = image.crop(img, 10, 10, 490, 490)
	    local img = image.scale(img,img_size,img_size)
	    ibt[j] = img
	    il[j] = i
	    j=j+1; print(j)
	end
	for i=1,l do
	    local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	    local img = image.scale(img,500,500):double()
	    local img = image.crop(img, 10, 10, 500, 500)
	    local img = image.scale(img,img_size,img_size)
	    ibt[j] = img
	    il[j] = i
	    j=j+1; print(j)
	end
	for i=1,l do
	    local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	    local img = image.scale(img,500,500):double()
	    local img = image.crop(img, 10, 0, 500, 490)
	    local img = image.scale(img,img_size,img_size)
	    ibt[j] = img
	    il[j] = i
	    j=j+1; print(j)
	end
	for i=1,l do
	    local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	    local img = image.scale(img,500,500):double()
	    local img = image.crop(img, 0, 10, 480, 500)
	    local img = image.scale(img,img_size,img_size)
	    ibt[j] = img
	    il[j] = i
	    j=j+1; print(j)
	end
	for i=1,l do
	    local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	    local img = image.scale(img,500,500):double()
	    local img = image.crop(img, 0, 20, 490, 490)
	    local img = image.scale(img,img_size,img_size)
	    ibt[j] = img
	    il[j] = i
	    j=j+1; print(j)
	end

	for i=1,l do
	    local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	    local img = image.scale(img,500,500):double()
	    local img = image.crop(img, 0, 0, 480, 490)
	    local img = image.scale(img,img_size,img_size)
	    ibt[j] = img
	    il[j] = i
	    j=j+1; print(j)
	end


	-- lighting
	n = {2,6,10}
	for k=1,3 do
	    b = torch.randn(500,500)*n[k]
	    for i=1,l do
	        local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	        local img = image.scale(img,500,500):double()
	        img[1] = img[1] +b
	        local img = image.scale(img,img_size,img_size)
	        ibt[j] = img
	        il[j] = i
	        j=j+1; print(j)
	    end
	    for i=1,l do
	        local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	        local img = image.scale(img,500,500):double()
	        local img = image.crop(img, 0, 0, 490, 490)
	        local img = image.scale(img,500,500):double()
	        img[1] = img[1] +b
	        local img = image.scale(img,img_size,img_size)
	        ibt[j] = img
	        il[j] = i
	        j=j+1; print(j)
	    end
	    for i=1,l do
	        local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	        local img = image.scale(img,500,500):double()
	        local img = image.crop(img, 10, 10, 490, 490)
	        local img = image.scale(img,500,500):double()
	        img[1] = img[1] +b
	        local img = image.scale(img,img_size,img_size)
	        ibt[j] = img
	        il[j] = i
	        j=j+1; print(j)
	    end
	    for i=1,l do
	        local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	        local img = image.scale(img,500,500):double()
	        local img = image.crop(img, 10, 10, 500, 500)
	        local img = image.scale(img,500,500):double()
	        img[1] = img[1] +b
	        local img = image.scale(img,img_size,img_size)
	        ibt[j] = img
	        il[j] = i
	        j=j+1; print(j)
	    end
	    for i=1,l do
	        local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	        local img = image.scale(img,500,500):double()
	        local img = image.crop(img, 10, 0, 500, 480)
	        local img = image.scale(img,500,500):double()
	        img[1] = img[1] +b
	        local img = image.scale(img,img_size,img_size)
	        ibt[j] = img
	        il[j] = i
	        j=j+1; print(j)
	    end
	    for i=1,l do
	        local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	        local img = image.scale(img,500,500):double()
	        local img = image.crop(img, 0, 10, 480, 500)
	        local img = image.scale(img,500,500):double()
	        img[1] = img[1] +b
	        local img = image.scale(img,img_size,img_size)
	        ibt[j] = img
	        il[j] = i
	        j=j+1; print(j)
	    end
	    for i=1,l do
	        local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	        local img = image.scale(img,500,500):double()
	        local img = image.crop(img, 0, 20, 490, 480)
	        local img = image.scale(img,500,500):double()
	        img[1] = img[1] +b
	        local img = image.scale(img,img_size,img_size)
	        ibt[j] = img
	        il[j] = i
	        j=j+1; print(j)
	    end
	    for i=1,l do
	        local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	        local img = image.scale(img,500,500):double()
	        local img = image.crop(img, 20, 20, 480, 480)
	        local img = image.scale(img,500,500):double()
	        img[1] = img[1] +b
	        local img = image.scale(img,img_size,img_size)
	        ibt[j] = img
	        il[j] = i
	        j=j+1; print(j)
	    end
	    for i=1,l do
	        local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	        local img = image.scale(img,500,500):double()
	        local img = image.crop(img, 20, 0, 500, 480)
	        local img = image.scale(img,500,500):double()
	        img[1] = img[1] +b
	        local img = image.scale(img,img_size,img_size)
	        ibt[j] = img
	        il[j] = i
	        j=j+1; print(j)
	    end

	end

end


imgdat = {data=ibt, label=il, size = j-1}
torch.save('size_'..img_size..'.dat',imgdat)
return imgdat
