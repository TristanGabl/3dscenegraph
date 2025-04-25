
import json
import os
import pandas as pd
import numpy as np


# out = "label_mapping"
mapping = pd.read_csv("helper_repos/ScanNet/scans/scannetv2-labels.combined.tsv", sep='\t')
scannet_id_to_name = json.load(open("label_mapping/scannet_id_to_name.json", 'r'))
scannet_name_to_color = json.load(open("label_mapping/scannet_to_color.json", 'r'))
coco_id_to_name = json.load(open("label_mapping/coco_id_to_name.json", 'r'))
coco_name_to_scannet_name = json.load(open("label_mapping/coco_name_to_scannet_name.json", 'r'))


# create scannet_id_to_name mapping
scannet_id_to_name = mapping.set_index('nyu40id')['nyu40class'].to_dict() 
scannet_id_to_name[-1] = "background"

scannet_id_to_name = dict(sorted(scannet_id_to_name.items()))

with open("label_mapping/scannet_id_to_name.json", "w") as json_file:
    json.dump(scannet_id_to_name, json_file, indent=4)


# create a mapping from coco_name_to_scannet_name (done manually)
# coco_name_to_scannet_name = {}
# # Use intersection for coco_id_to_name and scannet_id_to_name as a starting point
# for coco_id, coco_name in coco_id_to_name.items():
#     found_direct_match = False
#     for scannet_id, scannet_name in scannet_id_to_name.items():
#         if coco_name.lower() == scannet_name.lower():
#             coco_name_to_scannet_name[coco_name] = scannet_name
#             found_direct_match = True
#             break
#     if not found_direct_match:
#         coco_name_to_scannet_name[coco_name] = ""
# with open("label_mapping/coco_name_to_scannet_name.json", "w") as json_file:
#     json.dump(coco_name_to_scannet_name, json_file, indent=4)


# create a mapping from coco_id_to_scannet_id
scannet_name_to_id = {name: id for id, name in scannet_id_to_name.items()}

coco_id_to_scannet_id = {}
for coco_id, coco_name in coco_id_to_name.items():
    scannet_id = scannet_name_to_id[coco_name_to_scannet_name[coco_name]]
    if scannet_id is not None:
        coco_id_to_scannet_id[coco_id] = scannet_id

# Export coco_id_to_scannet_id to JSON
with open("label_mapping/coco_id_to_scannet_id.json", "w") as json_file:
    # Find scannet names that are not mapped to any coco id
    print(f"Number of unique values in scannet_id_to_name: {len(set(coco_name_to_scannet_name.values()))}")
    print(f"Leftover scannet names: {set(coco_name_to_scannet_name.values())}")
    json.dump(coco_id_to_scannet_id, json_file, indent=4)


def rainbow_rgb(t):
    """
    t: array‐like of floats in [0,1]
    returns: array of shape (len(t),3) with RGB in [0,1]
    """
    t = np.asarray(t)
    # Angular frequency
    ω = 2*np.pi
    # Phase shifts: 0, 120°, 240°
    r = 0.5 + 0.5 * np.sin(ω * t + 0)
    g = 0.5 + 0.5 * np.sin(ω * t - 2*np.pi/3)
    b = 0.5 + 0.5 * np.sin(ω * t - 4*np.pi/3)
    return [r, g, b]
scannet_id_to_color = {}
spacing = 1.0 / len(scannet_id_to_name)
# Define a color mapping for the 40 classes
for i, key in enumerate(scannet_id_to_name.keys()):
    if key == -1:
        # background always black
        scannet_id_to_color[key] = [0, 0, 0]
        continue
    
    # Generate a color using the rainbow function
    color = np.array(rainbow_rgb(i * spacing))
    color = color * 255  # Scale to [0, 255]
    color = color.tolist()  # Convert to list
    scannet_id_to_color[key] = [int(c) for c in color]
    
with open("label_mapping/scannet_id_to_color.json", "w") as json_file:
    json.dump(scannet_id_to_color, json_file, indent=4)


# create mapping from coco_id_to_name to simplified names
coco_name_to_name_simplified = {}
for coco_id, name in coco_id_to_name.items():
    simpler_name = name
    if len(name.split("-")) > 1:
        simpler_name = name.split("-")[0]
    
    coco_name_to_name_simplified[name] = simpler_name

# Export coco_name_to_name_simplified to JSON
with open(os.path.join("label_mapping/coco_name_to_name_simplified.json"), "w") as json_file:
    json.dump(coco_name_to_name_simplified, json_file, indent=4)

# _3dssg_relationships = []
# with open("helper_repos/3RScan/3DSSG/relationships.txt", "r") as file:
#     for line in file:
#         _3dssg_relationships.append(line.strip())
# _3dssg_relationships_to_corrected_relationships = {}
# for rel in _3dssg_relationships:
#     if rel.split("-")[0] == "same":
#         rel = "same"
    


