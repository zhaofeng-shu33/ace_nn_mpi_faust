# author: zhaofeng-shu33
# file: load image file and crop it to suitable size
import os
import pdb
import numpy as np
from PIL import Image
def get_image_bounding_box(img):
    # (x, y, x_b, y_b) where (x, y): upper-left corner
    # (
    im2 = img.convert('1') # boolean
    a = np.array(im2)
    a1 = np.where(np.sum(a, axis=0)!=a.shape[0])    
    a2 = np.where(np.sum(a, axis=1)!=a.shape[1])        
    pdb.set_trace()    
    x = a1[0][0]
    y = a2[0][0]
    x_b = a1[0][-1]
    y_b = a2[0][-1]
    return (x, y, x_b, y_b)
def save_images_to_npx():
    # sequence is important
    return
# iterate all images twice, the first time to calculate the largest width and height
# the second time to crop all images to suitable size
if __name__ == '__main__':
    max_width = 0
    max_height = 0
    cnt = 0
    for im_file in os.listdir('output'):
        im = Image.open(os.path.join('output', im_file))
        box = get_image_bounding_box(im)
        width = box[2] - box[0]
        height = box[3] - box[1]
        if(width > max_width):
            max_width = width
        if(height > max_height):
            max_height = height
        cnt +=1
        if(cnt % 100 == 0):
            print('%d/1000'%cnt)
    print('max width: %d' % max_width)
    print('max height: %d' % max_height)    
    # region = im.crop(box)
    # region.show()