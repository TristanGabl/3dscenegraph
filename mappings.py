
import json
import pandas as pd

mapping = pd.read_csv("ScanNet/scans/scannetv2-labels.combined.tsv", sep='\t')

scannet_id_to_name = json.load(open("scannet_id_to_name.json", 'r'))
name_to_color = json.load(open("scannet_to_color.json", 'r'))
coco_id_to_name = json.load(open("coco_id_to_name.json", 'r'))
coco_name_to_scannet_name = json.load(open("coco_name_to_scannet_name.json", 'r'))

# Create reverse mapping of scannet_id_to_name
name_to_scannet_id = {name: scannet_id for scannet_id, name in scannet_id_to_name.items()}

# Create a mapping from coco_id to scannet_id
coco_id_to_scannet_id = {}
coco_id_to_scannet_id["-1"] = "0"
for coco_id, name in coco_id_to_name.items():
    if name in coco_name_to_scannet_name:
        scannet_name = coco_name_to_scannet_name[name]
        scannet_id = name_to_scannet_id.get(scannet_name)
        coco_id_to_scannet_id[coco_id] = scannet_id
    else:
        print(f"Warning: {name} not found in name_to_scannet_id mapping.")

# Export coco_id_to_scannet_id to JSON
with open("coco_id_to_scannet_id.json", "w") as json_file:
    json.dump(coco_id_to_scannet_id, json_file, indent=4)

name_to_color = json.load(open("scannet_to_color.json", 'r'))
scannet_id_to_color = {}
for scannet_id, name in scannet_id_to_name.items():
    if name in name_to_color:
        scannet_id_to_color[scannet_id] = name_to_color[name]
    else:
        print(f"Warning: {name} not found in name_to_color mapping.")

        # Export scannet_id_to_color to JSON
with open("scannet_id_to_color.json", "w") as json_file:
    json.dump(scannet_id_to_color, json_file, indent=4)
