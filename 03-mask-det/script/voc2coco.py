import os
import json
import xml.etree.ElementTree as ET

def voc2coco(input_folder,output_file):
    # # 定义输入和输出文件路径
    # input_folder = "train_annos"
    # output_file = "train.json"

    # 定义类别列表和映射表
    classes = ["no_mask", "mask"]
    category_ids = {name: i+1 for i, name in enumerate(classes)}

    # 定义 COCO 格式的字典
    coco_data = {
        "images": [],
        "categories": [{"id": i, "name": name} for name, i in category_ids.items()],
        "annotations": []
    }

    # 遍历所有 XML 文件
    for xml_file in os.listdir(input_folder):
        if not xml_file.endswith(".xml"):
            continue

        # 解析 XML 文件
        tree = ET.parse(os.path.join(input_folder, xml_file))
        root = tree.getroot()

        # 获取图像信息
        image_file = root.find("filename").text
        image_id = int(os.path.splitext(image_file)[0].split("_")[-1])
        image = {
            "id": image_id,
            "file_name": image_file,
            "width": int(root.find("size/width").text),
            "height": int(root.find("size/height").text)
        }
        coco_data["images"].append(image)

        # 获取所有对象信息
        for obj in root.findall("object"):
            category = obj.find("name").text
            bbox = obj.find("bndbox")
            x_min = float(bbox.find("xmin").text)
            y_min = float(bbox.find("ymin").text)
            x_max = float(bbox.find("xmax").text)
            y_max = float(bbox.find("ymax").text)

            # 构建 COCO 格式的标注
            annotation = {
                "id": len(coco_data["annotations"]) + 1,
                "image_id": image_id,
                "category_id": category_ids[category],
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)

    # 将 COCO 格式的数据保存到 JSON 文件
    with open(output_file, "w") as f:
        json.dump(coco_data, f)
    print('success！！！')

if __name__ == "__main__":
    voc2coco(input_folder = "mask_dataset/annotations/val_annos",    output_file = "mask_dataset/val.json")