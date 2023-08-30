import os
import numpy as np
from HoughTransform import Hough, Canny

source_path = 'C:/Users/Isabelle/Desktop/FilterImages/images'

for filename in os.listdir(source_path):
    source = source_path + '/'
    img_file = filename[:len(source_path)]
    edges = Canny(img_file,source,50,100)
    lines = Hough(source,img_file,edges,1,np.pi/180,60,50,5)
