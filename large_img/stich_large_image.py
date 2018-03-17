import pickle
from large_img import non_max_suppression
import cv2
import numpy as np
from data_prep.parseTaggingServiceXML import parse_xml_to_dict


def stich_large_image(large_image_folder, pickle_file, output_folder):

    IoU_thresh = 0.2

    with open(pickle_file, 'rb') as handle:
        b = pickle.load(handle)

    image_name = b['image_name']
    # image_name = "big_image_with_stride"
    bboxes = b['bounding_boxes']

    large_image_file = large_image_folder + "images/"+image_name+".jpg"
    xml_annoation_file = large_image_folder + "annotations/"+image_name+".xml"

    # large_image_folder = "D:/OneDrive/Documents/FruitSpec/datasets/Tagging Service/all_images/images/"
    # pickle_file = 'img_annotations_with_stride_256.pickle'
    # loading predicted bboxes and performing non max suppression

    img = cv2.imread(large_image_file)

    # # drawing bbox before nms
    # for bbox in bboxes:
    #     cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 3)

    bboxes_nms = non_max_suppression.non_max_suppression_fast(bboxes, IoU_thresh, max_boxes=1000)
    print('detections before non max suppression: {}'.format(len(bboxes)))
    print('detections: {}'.format(len(bboxes_nms)))

    # drawing bbox of predictions
    for bbox in bboxes_nms:
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)

    # parsing XML of ground truth
    bboxes_gt = parse_xml_to_dict(xml_annoation_file)
    bboxes_gt = bboxes_gt['bbox']

    # drawing bbox of ground truth
    for bbox in bboxes_gt:
        cv2.rectangle(img, (int(bbox['xmin']),int(bbox['ymin'])), (int(bbox['xmax']), int(bbox['ymax'])), (0, 255, 0), 2)


    ground_truth = len(bboxes_gt)
    detections = len(bboxes_nms)
    accuracy = round(len(bboxes_nms) / len(bboxes_gt)*100)
    text_label = 'detected fruits: {}, ground truth: {} | accuracy: {}'.format(detections, ground_truth, accuracy)
    text_org = (0, np.size(img, 0) - 10)
    cv2.putText(img, text_label, text_org, cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite(output_folder + "/" + image_name + 'after_nms.jpg', img)

    return bboxes_nms

# def add
