import numpy as np
import cv2
from parseTaggingServiceXML import parse_xml_to_dict
from lxml import etree as ET
import os
import progressbar
import time
import glob

def createXML(sub_img_annotation, sub_image_ind_row, sub_image_ind_col):
    """
    function creates annotation file similar to original xml file.
    :param sub_img_annotation_dict: dict of subimage bbox, file name and folder name
    :param sub_image_ind_row: row index of sub image in big image
    :param sub_image_ind_col: col index of sub image in big image
    :return: xml file of data
    """
    # creating xml elements
    annotation = ET.Element('annotation')
    tree = ET.ElementTree(annotation)

    folder = ET.SubElement(annotation,'folder').text = sub_img_annotation['folder']
    filename = ET.SubElement(annotation,'filename').text = sub_img_annotation['filename']
    size = ET.SubElement(annotation,'size')
    width = ET.SubElement(size,'width').text = str(sub_img_annotation['width'])
    height = ET.SubElement(size, 'height').text = str(sub_img_annotation['height'])
    depth = ET.SubElement(size, 'depth').text = str(sub_img_annotation['depth'])

    img_top_left_corner_x = ET.SubElement(annotation,'img_top_left_corner_x')
    img_top_left_corner_x0 = ET.SubElement(img_top_left_corner_x,'x0').text = str(sub_img_annotation['x_sub_image'])
    img_top_left_corner_col = ET.SubElement(img_top_left_corner_x, 'column_shift').text = str(sub_image_ind_col)

    img_top_left_corner_y = ET.SubElement(annotation,'img_top_left_corner_y')
    img_top_left_corner_y0 = ET.SubElement(img_top_left_corner_y, 'y0').text = str(sub_img_annotation['y_sub_image'])
    img_top_left_corner_row = ET.SubElement(img_top_left_corner_y, 'row_shift').text = str(sub_image_ind_row)


    for bbox_ind,bbox in enumerate(sub_img_annotation['bbox']):
        # creating tree for bbox
        object = ET.SubElement(annotation,'object')
        name = ET.SubElement(object,'name').text = bbox['class']
        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')

        #setting values of bbox
        xmin.text = str(bbox['xmin'])
        ymin.text = str(bbox['ymin'])
        xmax.text = str(bbox['xmax'])
        ymax.text = str(bbox['ymax'])

    # create a new XML file with the results
    return tree

def split_image_annotation(big_img_data, xTopLeft, yTopLeft, xBottomRight, yBottomRight):
    """

    :param big_img_data: annotation data of image before the split
    :param xTopLeft: coordinate of top left corner
    :param yTopLeft: coordinate of top left corner
    :param xBottomRight: coordinate of bottom right corner
    :param yBottomRight: coordinate of bottom right corner
    :return: splitting annotation to fit with the small images according to corner coordinates
    """

    # veryfing coordinates of coordners are sorted
    xTopLeft.sort()
    yTopLeft.sort()
    xBottomRight.sort()
    yBottomRight.sort()

    # coordinates of sub images in pixels of large image
    xTopLeft = np.array(xTopLeft)
    yTopLeft = np.array(yTopLeft)
    xBottomRight = np.array(xBottomRight)-1
    yBottomRight = np.array(yBottomRight)-1


    # creating list of dictionaries. indexing based on IndX, indY
    all_img_data = []
    for ind in range(len(yTopLeft)):
        data_dict = {}
        data_dict['bbox'] = []
        data_dict['folder'] = []
        data_dict['bbox'] = []
        data_dict['filename'] = []
        all_img_data.append([])
        for ind2 in range(len(xTopLeft)):
            all_img_data[ind].append({})


    ind = 0
    while ind<len(big_img_data['bbox']):
        bbox = big_img_data['bbox'][ind]
    # for ind, bbox in enumerate(big_img_data['bbox']):
        new_data = {}
        # all_img_data[indY][indX]['bbox'] = []
        # all_img_data[indY][indX]['folder'] = []
        # all_img_data[indY][indX]['folder'].append(big_img_data['folder'])

        # coordinate of bbox
        xmin = float(bbox['xmin'])
        ymin = float(bbox['ymin'])
        xmax = float(bbox['xmax'])
        ymax = float(bbox['ymax'])
        class_name = bbox['class']

        if ymax > max(yBottomRight):
            ymax = max(yBottomRight)
        if xmax > max(xBottomRight):
            xmax = max(xBottomRight)


        # width and height of sub image
        img_width = xBottomRight+1 - xTopLeft
        img_height = yBottomRight+1 - yTopLeft
        # width and height of bbox
        w = xmax - xmin
        h = ymax - ymin

        # over looking bbox which are too small
        # if min(h,w) < 20:
        #     ind = ind+1
        #     continue

        # center of bbox
        x0 = np.mean([xmin, xmax])
        y0 = np.mean([ymin, ymax])

        # ind to sub image that holds top left corner Top Left Corner

        indX_TopLeft = max(np.where(xmin >= xTopLeft)[0])
        indY_TopLeft = max(np.where(ymin >= yTopLeft)[0])

        # ind to sub image that holds Bottom Right corener
        try:
            indX_BottomRight = min(np.where(xmax <= xBottomRight)[0])
            if indX_BottomRight == -1:
                indX_BottomRight = len(yBottomRight)
        except Exception:
            indX_BottomRight = len(yBottomRight)
        try:
            indY_BottomRight = min(np.where(ymax <= yBottomRight)[0])
            if indY_BottomRight == -1:
                indY_BottomRight = len(yBottomRight)
        except Exception:
            indY_BottomRight = len(yBottomRight)

        # checking of bbox crosses sub image border:
        split_bbox_in_x = 0
        split_bbox_in_y = 0

        if indX_TopLeft != indX_BottomRight:
            split_bbox_in_x = 1
        if indY_TopLeft != indY_BottomRight:
            split_bbox_in_y = 1

        # only bbox that don't cross sub image are saved. if bbox crosses, it is split and will be analyzed in the end
        # splitting bbox in x
        if split_bbox_in_x:
            # bbox Left
            xmin_new_box = xmin
            ymin_new_box = ymin
            xmax_new_box = xBottomRight[indX_BottomRight-1]
            ymax_new_box = ymax
            big_img_data['bbox'].append({'xmin': xmin_new_box,
                                         'ymin': ymin_new_box,
                                         'xmax': xmax_new_box,
                                         'ymax': ymax_new_box,
                                         'class': class_name})
            # bbox Right
            xmin_new_box = xTopLeft[indX_TopLeft+1]
            ymin_new_box = ymin
            xmax_new_box = xmax
            ymax_new_box = ymax
            big_img_data['bbox'].append({'xmin': xmin_new_box,
                                         'ymin': ymin_new_box,
                                         'xmax': xmax_new_box,
                                         'ymax': ymax_new_box,
                                         'class': class_name})
            ind = ind+1
            continue    #bbox will be saved later (or will be split again in y if crossed border in x and y)

            # splitting bbox in both y
        if split_bbox_in_y:
            # bbox Top
            xmin_new_box = xmin
            ymin_new_box = ymin
            xmax_new_box = xmax
            ymax_new_box = yBottomRight[indY_BottomRight-1]
            big_img_data['bbox'].append({'xmin': xmin_new_box,
                                         'ymin': ymin_new_box,
                                         'xmax': xmax_new_box,
                                         'ymax': ymax_new_box,
                                         'class': class_name})
            # bbox Bottom
            xmin_new_box = xmin
            ymin_new_box = yTopLeft[indY_TopLeft+1]
            xmax_new_box = xmax
            ymax_new_box = ymax
            big_img_data['bbox'].append({'xmin': xmin_new_box,
                                         'ymin': ymin_new_box,
                                         'xmax': xmax_new_box,
                                         'ymax': ymax_new_box,
                                         'class': class_name})
            ind = ind + 1
            continue  # bbox will be saved later (will be split again in y if crossed border in x and y)


        xmin = max(0,xmin - xTopLeft[indX_TopLeft])
        ymin = max(0,ymin - yTopLeft[indY_TopLeft])

        xmax = min(xmax - xTopLeft[indX_TopLeft], img_width[indX_TopLeft])
        ymax = min(ymax - yTopLeft[indY_TopLeft], img_height[indY_TopLeft])


        new_bbox = {'class': class_name,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax}


        try:
            all_img_data[indY_TopLeft][indX_TopLeft]['folder'] = big_img_data['folder']
            all_img_data[indY_TopLeft][indX_TopLeft]['filename'] = big_img_data['filename'][:-4] + '_' + str(indY_TopLeft) + '_' + str(indX_TopLeft) + '.jpg'
            all_img_data[indY_TopLeft][indX_TopLeft]['x_sub_image'] = xTopLeft[indX_TopLeft]
            all_img_data[indY_TopLeft][indX_TopLeft]['y_sub_image'] = xTopLeft[indY_TopLeft]

            all_img_data[indY_TopLeft][indX_TopLeft]['width'] = xBottomRight[indX_BottomRight]-xTopLeft[indX_TopLeft]+1
            all_img_data[indY_TopLeft][indX_TopLeft]['height'] = yBottomRight[indY_BottomRight]-yTopLeft[indY_TopLeft]+1
            all_img_data[indY_TopLeft][indX_TopLeft]['depth'] = big_img_data['depth']

            all_img_data[indY_TopLeft][indX_TopLeft]['bbox'].append(new_bbox)


        except Exception:   # if get here, it means this is the first time indY,indX have been reached

            all_img_data[indY_TopLeft][indX_TopLeft]['filename'] = []
            all_img_data[indY_TopLeft][indX_TopLeft]['filename'] = big_img_data['filename'][:-4]+'_'+str(indY_TopLeft)+'_'+str(indX_TopLeft)+'.jpg'

            all_img_data[indY_TopLeft][indX_TopLeft]['width'] = []
            all_img_data[indY_TopLeft][indX_TopLeft]['height'] = []
            all_img_data[indY_TopLeft][indX_TopLeft]['depth'] = []

            all_img_data[indY_TopLeft][indX_TopLeft]['bbox'] = []
            all_img_data[indY_TopLeft][indX_TopLeft]['bbox'].append(new_bbox)

        ind = ind+1
    return  all_img_data

def split_image(img, xsteps, ysteps, phase = 0):
    """

    :param img: large image
    :param xsteps: step size to split in x
    :param ysteps: step size to split in y
    :param phase: the area of overlap between images
    :return: small images from split image
    """

    (rows, cols, depth) = np.shape(img)
    cols = list(range(cols))
    rows = list(range(rows))

    xTopLeft = cols[0:-1:xsteps] + cols[phase:-1:xsteps]
    yTopLeft = rows[0:-1:ysteps] + rows[phase:-1:ysteps]

    xTopLeft = list(set(xTopLeft))
    yTopLeft = list(set(yTopLeft))

    xTopLeft.sort()
    yTopLeft.sort()

    xBottomRight = np.add(xTopLeft, xsteps) #cols[xsteps:-1:xsteps] + cols[phase+xsteps:-1:xsteps]
    yBottomRight = np.add(yTopLeft, ysteps) #rows[ysteps:-1:ysteps] + rows[phase+ysteps:-1:ysteps]

    # while len(xBottomRight) < len(xTopLeft):
    #     xBottomRight.append(cols[-1])
    # while len(yBottomRight) < len(yTopLeft):
    #     yBottomRight.append(rows[-1])


    xBottomRight[np.where(xBottomRight > np.shape(img)[1])] = np.shape(img)[1]
    yBottomRight[np.where(yBottomRight > np.shape(img)[0])] = np.shape(img)[0]

    xBottomRight[-1] = np.shape(img)[1]
    yBottomRight[-1] = np.shape(img)[0]

    img_split = []
    counter = 0
    for rind, y0 in enumerate(yTopLeft):
        img_split_row = []
        for cind, x0 in enumerate(xTopLeft):
            img_split_row.append(img[yTopLeft[rind]:yBottomRight[rind], xTopLeft[cind]:xBottomRight[cind]])
        img_split.append(img_split_row)

    # img = cv2.resize(img, None, fx = 0.2, fy = 0.2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    return img_split, xTopLeft, yTopLeft, xBottomRight, yBottomRight

def create_new_data_from_big_image(img, annotation_data, out_folder, filename):
    """
    split big image and its annotations to smaller images with annotations
    :param: out_folder: parent folder where database is saved. will have two sub folders: "images", "annotations"
    :param: filename: name of image and annotation original file. must have same name.
    :param: img: big image to be split into sub images
    :param: annotation_data: annotations of img
    :return: saves sub images and their annotations
    """

    # folder = "D:/OneDrive/Documents/FruitSpec/datasets/Tagging Service/201217"
    # filename = "20171220_072604"
    # img_file_name = folder+"/images/"+filename+".jpg"

    # reading image and annotation data
    # img = cv2.imread(img_file_name)
    # annotation_data = folder + "/annotations/" + filename + ".xml"


    ysteps = 512
    xsteps = 512
    phase = 0
    sub_imgs, xTopLeft, yTopLeft, xBottomRight, yBottomRight = split_image(img, xsteps, ysteps, phase)

    # reading annotation data
    big_img_data = parse_xml_to_dict(annotation_data)

    # img = cv2.resize(img, None, fx=0.2, fy=0.2)
    # cv2.imshow('img', img)

    # splitting annotation of big image to xml for every sub image
    sub_img_annotation = split_image_annotation(big_img_data, xTopLeft, yTopLeft, xBottomRight, yBottomRight)
    # print('finished splitting annotation data')
    """
    split data has the following structure
    split_data[subImgIndY][subImgIndX]
        'folder'
        'filename' --> file of image
        'y_sub_image' --> row of sub image in big image
        'x_sub_image' --> column of sub image in big image
        'bbox':
            'xmin'
            'ymin'
            'xmax'
            'ymax'
    """

    # saving the sub images and annotations of sub images
    for rind in range(np.shape(sub_img_annotation)[0]):
        for cind in range(np.shape(sub_img_annotation)[1]):
            #creating and saving XML file of small image annotation
            try:
                xml_sub_img = createXML(sub_img_annotation[rind][cind], rind, cind)
                img = sub_imgs[rind][cind]

            except Exception:
                continue

            # saving image and XML
            cv2.imwrite((out_folder+"/images/"+filename+"_{}_{}.jpg".format(rind, cind)), img)
            xml_sub_img.write((out_folder+"/annotations/"+filename+"_{}_{}.xml".format(rind, cind)), pretty_print=True)

def parse_XML_folder(parent_folder):
    # parent_folder = "D:/OneDrive/Documents/FruitSpec/datasets/Tagging Service/201217"
    annotation_folder = parent_folder+"/annotations"
    images_folder = parent_folder+"/images"
    out_folder = parent_folder+"/sub_images"
    if os.path.isdir(out_folder) == False:
        os.mkdir(out_folder)
    try:
        os.mkdir(out_folder+ "/annotations")
        os.mkdir(out_folder+ "/images")
    except Exception:
        print('did not create new folder. /annotations; /images; folder already exist')

    bar = progressbar.ProgressBar(maxval=len(os.listdir(annotation_folder)))
    bar.start(max_value=len(os.listdir(annotation_folder)))
    bar_len = 0
    for filename in os.listdir(annotation_folder):
        bar_len = bar_len+1
        bar.update(bar_len)
        time.sleep(0.02)

        if filename.endswith(".xml"):
            try:
                ind = filename.rfind('.xml')
                filename = filename[:ind]
                annotation_xml_file = annotation_folder+"/"+filename+".xml"
                # finding image file without knowing the image file type (tiff/jpg/png...)
                for infile in glob.glob(images_folder+"/"+filename+'*'):
                    image_filename = infile

                # img = cv2.imread(images_folder+"/"+filename+".jpg")
                img = cv2.imread(image_filename)
                create_new_data_from_big_image(img, annotation_xml_file, out_folder, filename)
            except Exception as e:
                print(e)
                continue
    bar.finish()