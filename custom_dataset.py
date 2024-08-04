from torch.utils.data import Dataset
from PIL import Image
import os
import glob
from torchvision import transforms
import xml.etree.ElementTree as ET

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.image_path = f'{root_dir}/VOC2012/JPEGImages'
        self.annotation_path = f'{root_dir}/VOC2012/Annotations'
        self.classes_path = f'{root_dir}/VOC2012/ImageSets/Main'
        self.transform = transform
        self.classes = self.get_classes()
        self.images = sorted([os.path.join(self.image_path, file) for file in os.listdir(self.image_path)])
        self.annotations = sorted([os.path.join(self.annotation_path, file) for file in os.listdir(self.annotation_path)])

    def get_classes(self):
        path = f'{self.classes_path}/*_*.txt'
        list = [os.path.basename(file).split('_')[0] for file in glob.glob(path)]
        return sorted(set(list))

    def __len__(self):
        return len(self.images)

    def __getbbox__(self, idx, scale):
        tree = ET.parse(self.annotations[idx])
        root = tree.getroot()
        objects_info = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            objects_info.append({
                'name': name,
                'x': int((xmin+xmax)/2*scale[0]),
                'y': int((ymin+ymax)/2*scale[1]),
                'width': int((xmax-xmin)*scale[0]),
                'height': int((ymax-ymin)*scale[1]),
            })
        return objects_info

    def __getann__(self, bboxes):
        annotations = [[[0 for _ in range(30)] for _ in range(7)] for _ in range(7)]
        for bbox in bboxes:
            list = [1, (bbox['x']-int(bbox['x']/64)*64)/64, (bbox['y']-int(bbox['y']/64)*64)/64, bbox['width']/448, bbox['height']/448,0,0,0,0,0]
            zero_list = [0]*len(self.classes)
            index = self.classes.index(bbox['name'])
            zero_list[index] = 1
            list.extend(zero_list)
            annotations[int(bbox['y']/64)][int(bbox['x']/64)]=list
        return annotations

    def __getitem__(self, idx):
        scale = (1,1)
        img_path = self.images[idx]
        image = Image.open(img_path)
        tr_image = self.transform(image)
        scale = (tr_image.shape[1]/image.size[0],tr_image.shape[2]/image.size[1])
        bounding_boxes = self.__getbbox__(idx, scale)
        annotations = self.__getann__(bounding_boxes)
        return tr_image, annotations
