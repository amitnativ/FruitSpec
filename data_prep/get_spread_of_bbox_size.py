"""
display the spread of bbox sizes of original annotation data files.
this is important for selecting the size of anchors in faster-rcnn
"""

import numpy as np
import matplotlib.pyplot as plt


filename = 'D:\\OneDrive\\Documents\\FruitSpec\\PYTHON\\FruitSpec_FasterRCNN\\annotationTextFileTaggingService.txt'
# going over all bboxes in all data from text file
xmin = []
ymin = []
xmax = []
ymax = []

with open(filename) as f:
    for line in f:
        try:
            (img_file, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_class) = line[:-1].split(",")
            xmin.append(float(bbox_xmin))
            ymin.append(float(bbox_ymin))
            xmax.append(float(bbox_xmax))
            ymax.append(float(bbox_ymax))
        except Exception as e:
            print(e)

xmin = np.array(xmin)
ymin = np.array(ymin)
xmax = np.array(xmax)
ymax = np.array(ymax)

w = xmax-xmin
h = ymax-ymin

area = w*h
bbox_size = np.sqrt(area)

# plotting histogram
bins = np.linspace(0,150,16)
print(bins)
plt.hist(bbox_size, bins, align='left', rwidth=0.5)
plt.title("bbox size")
plt.xlabel("size")
plt.ylabel("count")
plt.show()


