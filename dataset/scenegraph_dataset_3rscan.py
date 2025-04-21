import json
import os
import glob
from utils.scenegraph3d_objects import Objects

# Define file paths
base_path = "3RScan/"
semseg_files = glob.glob(os.path.join(base_path, "data", "3RScan", "*", "semseg.v2.json"))
objects_file = os.path.join(base_path, "3DSSG/objects.json")
relationships_file = os.path.join(base_path, "3DSSG/relationships.json")

objects_data = None
relationships_data = None


# Read objects.json
with open(objects_file, 'r') as f:
    objects_data = json.load(f)

# Read relationships.json
with open(relationships_file, 'r') as f:
    relationships_data = json.load(f)


# Process each semseg file
scan_id_to_objects = {}
for semseg_file in semseg_files:
    with open(semseg_file, 'r') as f:
        semseg_data = json.load(f)
    
    # Extract object centroids
    for seg_group in semseg_data.get("segGroups", []):
        if "obb" in seg_group and "centroid" in seg_group["obb"]:
            scan_id = semseg_data.get("scan_id", "unknown_scan_id")
            if scan_id not in scan_id_to_objects:
                scan_id_to_objects[scan_id] = []
            # scan_id_to_objects[scan_id].append({
            #     "id": seg_group["id"],
            #     "label": seg_group.get("label"),
            #     "centroid": seg_group["obb"]["centroid"],
            #     "size": seg_group["obb"]["axesLengths"]
            # })
            # Create an object instance
            obj = Objects(
                name=seg_group.get("label"),
                object_id=seg_group["id"],
                class_id=-1,  # Assuming class_id is not provided in the JSON
                x=seg_group["obb"]["centroid"][0],
                y=seg_group["obb"]["centroid"][1],
                z=seg_group["obb"]["centroid"][2],
                size_x=seg_group["obb"]["axesLengths"][0],
                size_y=seg_group["obb"]["axesLengths"][1],
                size_z=seg_group["obb"]["axesLengths"][2]
            )
            scan_id_to_objects[scan_id].append(obj)

# Extract relationships for each scan
scan_id_to_relationships = {}
for scan_data in relationships_data.get("scans", []):
    scan_id = scan_data.get("scan")
    if scan_id not in scan_id_to_relationships:
        scan_id_to_relationships[scan_id] = []
    for relationship in scan_data.get("relationships", []):
        scan_id_to_relationships[scan_id].append({
            "source": relationship[0],
            "target": relationship[1],
            "type": relationship[3]
        })


        # Combine scan_id_to_objects and scan_id_to_relationships
scan_id_to_combined = {}
for scan_id in set(scan_id_to_objects.keys()).intersection(scan_id_to_relationships.keys()):
    scan_id_to_combined[scan_id] = {
        "objects": scan_id_to_objects.get(scan_id, []),
        "relationships": scan_id_to_relationships.get(scan_id, [])
    }

out_path = os.path.join(base_path, "relationship_training_data.jsonl")
with open(out_path, 'w', encoding='utf-8') as f:
    for scans in scan_id_to_combined.values():
        objs = scans["objects"]
        for rel in scans["relationships"]:
            # find source/target object instances
            src = next((o for o in objs if o.object_id == rel["source"]), None)
            tgt = next((o for o in objs if o.object_id == rel["target"]), None)
            if not src or not tgt or src == tgt:
                continue

            record = {
                "input": {
                    "source": {
                        "id":       src.object_id,
                        "name":     src.name,
                        "x":        src.x,
                        "y":        src.y,
                        "z":        src.z,
                        "size_x":   src.size_x,
                        "size_y":   src.size_y,
                        "size_z":   src.size_z,

                    },
                    "target": {
                        "id":       tgt.object_id,
                        "name":     tgt.name,
                        "x":        tgt.x,
                        "y":        tgt.y,
                        "z":        tgt.z,
                        "size_x":   tgt.size_x,
                        "size_y":   tgt.size_y,
                        "size_z":   tgt.size_z,
                    }
                },
                "output": rel["type"]
            }
            f.write(json.dumps(record) + "\n")

print(f"Wrote per‐pair JSONL to {out_path}")
                



