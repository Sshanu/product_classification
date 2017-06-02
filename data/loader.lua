require 'image'

img_size = tonumber(arg[1])
l = tonumber(arg[2])

ibt=torch.ByteTensor(l*4,3,img_size,img_size)
il=torch.ByteTensor(l*4)
j = 1
s = {'croped','diff','difflight','light'}
for x=1,4 do
  for i=1,l do
	    local img = image.load('images/'..s[x]..'/img_' .. tostring(i) .. '.jpg',3,'byte')
	    local img = image.scale(img,img_size,img_size)
	    ibt[j] = img
		  il[j] = i
	    j=j+1
	end
end

imgdat = {data=ibt, label=il, size = j-1}
torch.save('basic_size_'..img_size..'.dat',imgdat)
return imgdat
