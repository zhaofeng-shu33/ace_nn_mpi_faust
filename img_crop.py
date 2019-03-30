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
    x_middle = (a1[0][0] + a1[0][-1])/2
    if(x_middle - 370 < a1[0][0]):
        x = x_middle - 370
    else:
        x = a1[0][0]
    if(x_middle + 370 > a1[0][-1]):
        x_b = x_middle + 370
    else:
        x_b = x + 740
    return (x, 0, x_b, 1054)
def get_boolean_array(img):
    im2 = img.convert('1') # boolean
    a = np.array(im2)
    a1 = np.where(np.sum(a, axis=0)!=a.shape[0])    
    x_middle = int((a1[0][0] + a1[0][-1])/2)
    if(x_middle - 370 < a1[0][0]):
        x = x_middle - 370
    else:
        x = a1[0][0]
    if(x_middle + 370 > a1[0][-1]):
        x_b = x_middle + 370
    else:
        x_b = x + 740
    return a[:,x:x_b]
# iterate all images twice, the first time to calculate the largest width and height
# the second time to crop all images to suitable size
if __name__ == '__main__':
    max_width = 0
    max_height = 0
    cnt = 0
    pose = np.zeros([100,1054,740],dtype=np.bool)
    for pose_id in range(10):
        for person_id in range(10):
            for angle_id in range(10):
                img_file = '%d_%d_%d.jpg'%(person_id, pose_id, angle_id)
                im = Image.open(os.path.join('output', img_file))    
                pose_offset = person_id * 10 + angle_id
                pose[pose_offset,:,:] = get_boolean_array(im)
        np.save(os.path.join('output','%d.npx' % pose_id), pose)
        print('%d/10'%pose_id)
