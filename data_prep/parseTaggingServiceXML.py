"""
Created on Sun Dec 24 14:36:49 2017
@author: Amit Nativ

Parsing XML files
"""

# %% Imports...
import xml.etree.ElementTree as ET
import os
import progressbar
import cv2

# %% intializing
def parse_xml_to_dict(filename):

    classes_count = {}  # classes count is not a dictionary
    ind = filename.find('annotations')
    element_folder = filename[:ind-1]
    try:
        # %% Parsing
        # print('parsing ', filename, '...')
        tree = ET.parse(filename)
        element = tree.getroot()  # node to XML tree base

        # element_folder = element.find('folder').text  # text of folder path
        element_filename = element.find('filename').text  # text of image filename
        ind = filename.rfind('/')
        ind1 = filename.find('.xml')
        # element_filename = filename[ind+2:ind1]+'.jpg'
        # element_segmented = bool(element.find('segmented').text)  # stated if image was segmented
        element_size = element.find('size')  # node to image size
        try:
            width = int(element_size.find('width').text)
            height = int(element_size.find('height').text)
            depth = int(element_size.find('depth').text)

        except Exception:
            width = []
            height = []
            depth = []

        ind = filename.find(element_folder)
        # adding data above to data structure
        annotation_data = {'folder': filename[:ind]+element_folder+'/images',
                           'filename': element_filename,
                           'width': width,
                           'height': height,
                           'depth': depth}
        annotation_data['bbox'] = []

        # %% iterate over all objects in image: get box and class
        # nodes to all objects in the image
        # each object is then parsed by itself
        element_objs = element.findall('object')  # nodes of all obj in XML

        for element_obj in element_objs:
            class_name = element_obj.find('name').text
            # checking if class_name was already detected
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] = classes_count[class_name] + 1

            obj_bbox = element_obj.find('bndbox')  # node to object bbox elements
            # parsing coordinates of images
            xmin = int(round(float(obj_bbox.find('xmin').text)))
            xmax = int(round(float(obj_bbox.find('xmax').text)))
            ymin = int(round(float(obj_bbox.find('ymin').text)))
            ymax = int(round(float(obj_bbox.find('ymax').text)))

            # adding all the collected data together
            annotation_data['bbox'].append(
                {'class': class_name, 'xmin': xmin,
                 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

    except Exception as e:
        print('')
        print('Exception: {}, at {}'.format(e, filename))
        annotation_data = []

    return annotation_data


def parse_XML_folder(folder, create_text_file = True):
    """
    function parses all XML files in folder.
    as default will also create a text file in the form of the Kitti open source Frcnn
    :param folder: folder of XML files
    :param create_text_file: create text file to fit with open source text file format
    :return: a list containing annotations of all files in XML folder
    """
    all_data = []
    bar = progressbar.ProgressBar(maxval=len(os.listdir(folder))).start()
    bar_len = 0
    for filename in os.listdir(folder):
        bar_len = bar_len+1
        bar.update(bar_len)
        if filename.endswith(".xml"):
            annotation_data = parse_xml_to_dict(folder+'/'+filename)
            img = cv2.imread(annotation_data['folder'] + '/' + annotation_data['filename'][:-3] + 'tif')
            for bbox in annotation_data['bbox']:
                cv2.rectangle(img,(bbox['xmin'],bbox['ymin']),(bbox['xmax'],bbox['ymax']),(0,0,255),2)

            # img = cv2.resize(img,None,fx=0.5,fy=0.5)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)

            all_data.append(annotation_data)

    bar.finish()

    if create_text_file == True:
        create_annotation_text_file(all_data)


    return all_data

def create_annotation_text_file(all_data):
    """
    function creates text file in following format.
    for each bbox in each file:
    (file,xmin,ymin,xmax,ymax,class)
    :param all_data: annoation data of all XML files in original folder
    :return: creates text file called annotation_data.txt
    """
    folder = all_data[0]['folder'][:-7]
    # creating text file for writing
    txtfile = open(folder + "/annotation_text_file.txt", "a")

    for img_ind, img_data in enumerate(all_data):
        try:
            print(img_data['filename'])
            img_file = img_data['folder']+'/'+img_data['filename']
            for bbox_ind, bbox in enumerate(img_data['bbox']):
                xmin = bbox['xmin']
                ymin = bbox['ymin']
                xmax = bbox['xmax']
                ymax = bbox['ymax']
                bbox_class = bbox['class']

                if img_file.find('080755') > 0:
                    print(img_file)

                txtfile.write(img_file + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + bbox_class)
                txtfile.write('\n')

        except Exception as e:
            print('')
            print(all_data[img_ind])
            print('Exception: {}'.format(e))
            txtfile.write('\n')

    #closing text file
    txtfile.close()