import xml.etree.ElementTree as ET
import os
import json

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = 0
image_id = 0
annotation_id = 0

classes= {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3,
                         'bottle':4, 'bus':5, 'car':6, 'cat':7, 'chair':8,
                         'cow':9, 'diningtable':10, 'dog':11, 'horse':12,
                         'motorbike':13, 'person':14, 'pottedplant':15,
                         'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

coco['categories'] = [{"supercategory":"none","id":1,"name":"aeroplane"},
		        {"supercategory":"none","id":2,"name":"bicycle"},
		        {"supercategory":"none","id":3,"name":"bird"},
		        {"supercategory":"none","id":4,"name":"boat"},
			{"supercategory":"none","id":5,"name":"bottle"},
			{"supercategory":"none","id":6,"name":"bus"},
			{"supercategory":"none","id":7,"name":"car"},
			{"supercategory":"none","id":8,"name":"cat"},
			{"supercategory":"none","id":9,"name":"chair"},
		        {"supercategory":"none","id":10,"name":"cow"},
		        {"supercategory":"none","id":11,"name":"diningtable"},
		        {"supercategory":"none","id":12,"name":"dog"},
			{"supercategory":"none","id":13,"name":"horse"},
			{"supercategory":"none","id":14,"name":"motorbike"},
			{"supercategory":"none","id":15,"name":"person"},
			{"supercategory":"none","id":16,"name":"pottedplant"},
			{"supercategory":"none","id":17,"name":"sheep"},
			{"supercategory":"none","id":18,"name":"sofa"},
			{"supercategory":"none","id":19,"name":"train"},
			{"supercategory":"none","id":20,"name":"tvmonitor"},
		      ]

def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = classes[name]#category_item_id
    category_item['name'] = name
    #coco['categories'].append(category_item)
    print(name + '  ' + str(classes[name]))
    category_set[name] = category_item_id
    return category_item_id

def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    #left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    #right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    #right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def parseXmlFiles(xml_path, _mode_path):
    XMLFiles = list()
    with open(_mode_path, "rb") as f:
	for line in f:
		fileName = line.strip()
		#print fileName
		XMLFiles.append(fileName + ".xml")

    for f in XMLFiles: #os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue
        
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)
        #print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        #elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None
            
            if elem.tag == 'folder':
                continue
            
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')
                
            #add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    #print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name)) 
            #subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox ['xmin'] = None
                bndbox ['xmax'] = None
                bndbox ['ymin'] = None
                bndbox ['ymax'] = None
                
                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        #print(object_name + '   ' + str(category_set))
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                #option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(float(option.text))

                #only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    #x
                    bbox.append(bndbox['xmin'])
                    #y
                    bbox.append(bndbox['ymin'])
                    #w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    #h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    #print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox )

if __name__ == '__main__':
    main_path="./ImageSets/Main"
    trainFile = main_path + '/' + 'train.txt'
    valFile = main_path + '/' + 'val.txt'
    testFile = main_path + '/' + 'test.txt'



    xml_path = './Annotations_xml_org'
    json_file = 'voc_2007_train.json'
    parseXmlFiles(xml_path, trainFile)
    json.dump(coco, open(json_file, 'w'))



    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    category_set = dict()
    image_set = set()
    category_item_id = 0
    image_id = 0
    annotation_id = 0
    coco['categories'] = [{"supercategory":"none","id":1,"name":"aeroplane"},
		        {"supercategory":"none","id":2,"name":"bicycle"},
		        {"supercategory":"none","id":3,"name":"bird"},
		        {"supercategory":"none","id":4,"name":"boat"},
			{"supercategory":"none","id":5,"name":"bottle"},
			{"supercategory":"none","id":6,"name":"bus"},
			{"supercategory":"none","id":7,"name":"car"},
			{"supercategory":"none","id":8,"name":"cat"},
			{"supercategory":"none","id":9,"name":"chair"},
		        {"supercategory":"none","id":10,"name":"cow"},
		        {"supercategory":"none","id":11,"name":"diningtable"},
		        {"supercategory":"none","id":12,"name":"dog"},
			{"supercategory":"none","id":13,"name":"horse"},
			{"supercategory":"none","id":14,"name":"motorbike"},
			{"supercategory":"none","id":15,"name":"person"},
			{"supercategory":"none","id":16,"name":"pottedplant"},
			{"supercategory":"none","id":17,"name":"sheep"},
			{"supercategory":"none","id":18,"name":"sofa"},
			{"supercategory":"none","id":19,"name":"train"},
			{"supercategory":"none","id":20,"name":"tvmonitor"},
		      ]
    json_file = 'voc_2007_val.json'
    parseXmlFiles(xml_path, valFile)
    json.dump(coco, open(json_file, 'w'))


    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    category_set = dict()
    image_set = set()
    category_item_id = 0
    image_id = 0
    annotation_id = 0
    coco['categories'] = [{"supercategory":"none","id":1,"name":"aeroplane"},
		        {"supercategory":"none","id":2,"name":"bicycle"},
		        {"supercategory":"none","id":3,"name":"bird"},
		        {"supercategory":"none","id":4,"name":"boat"},
			{"supercategory":"none","id":5,"name":"bottle"},
			{"supercategory":"none","id":6,"name":"bus"},
			{"supercategory":"none","id":7,"name":"car"},
			{"supercategory":"none","id":8,"name":"cat"},
			{"supercategory":"none","id":9,"name":"chair"},
		        {"supercategory":"none","id":10,"name":"cow"},
		        {"supercategory":"none","id":11,"name":"diningtable"},
		        {"supercategory":"none","id":12,"name":"dog"},
			{"supercategory":"none","id":13,"name":"horse"},
			{"supercategory":"none","id":14,"name":"motorbike"},
			{"supercategory":"none","id":15,"name":"person"},
			{"supercategory":"none","id":16,"name":"pottedplant"},
			{"supercategory":"none","id":17,"name":"sheep"},
			{"supercategory":"none","id":18,"name":"sofa"},
			{"supercategory":"none","id":19,"name":"train"},
			{"supercategory":"none","id":20,"name":"tvmonitor"},
		      ]
    json_file = 'voc_2007_test.json'
    parseXmlFiles(xml_path, testFile)
    json.dump(coco, open(json_file, 'w'))
