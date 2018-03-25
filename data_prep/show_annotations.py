# %% Imports...
import xml.etree.ElementTree as ET
import os
import cv2

def draw_XML_annotations(folder, filename):
   """
    function parses all XML files in folder.
    as default will also create a text file in the form of the Kitti open source Frcnn
    :param folder: folder of XML files
    :param filename: of xml file contating annotations to display
    :return: a list containing annotations of all files in XML folder
    """
   from data_prep import parseTaggingServiceXML

   all_data = []
   annotation_data = parseTaggingServiceXML.parse_xml_to_dict(folder + '/' + filename)
   img = cv2.imread(annotation_data['folder'] + '/' + annotation_data['filename'])
   for bbox in annotation_data['bbox']:
       cv2.rectangle(img, (bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax']), (0, 0, 255), 1)

   img = cv2.resize(img, None, fx=0.8, fy=0.8)
   cv2.imshow('img', img)
   cv2.waitKey(0)

   all_data.append(annotation_data)
   return all_data
