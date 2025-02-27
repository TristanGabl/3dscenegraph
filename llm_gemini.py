from google import genai
import json
import numpy as np

client = genai.Client(api_key="AIzaSyB9mTbKNa5SrHEvLK4HzdACiX2-VM4iNF0")

with open('/teamspace/studios/this_studio/3dscenegraph/output/apple_banana_scan/result/objects.json', 'r') as file:
    context = json.load(file)

# for obj in context:
#     response = client.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=f"context: {obj['name']} is at position {obj['position']}; given the context state what object is at position . give 1 sentence and nothing else, just say the fact",
#     )

# response = client.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=f"context: Apple 0 is at position (x,y,z) = (0,0,1.2); it has a neighbor dinner-table 0 at position (x,y,z) = (0,0,1); given this context and logic give a spacial relationship between the two objects. give 1 sentence and nothing else, just say the fact",
#     )


edges = [["" for i in range(len(context))] for j in range(len(context))]

# for i, obj in enumerate(context):
#     for j, neighbor in enumerate(obj['neighbors']):
#         neighbor_index = next((index for (index, d) in enumerate(context) if d["name"] == neighbor), None)
#         if neighbor_index is not None:
#             edges[i][neighbor_index] = 1

for i, obj in enumerate(context):
    for neighbor_id in obj['neighbors']:
        pass
        neighbor_obj = context[neighbor_id]
        prompt = f"context: {obj['name']} is at position (x,y,z) = ({obj['x']},{obj['y']},{obj['x']}). The object has a {neighbor_obj['name']} at position (x,y,z) = ({neighbor_obj['x']},{neighbor_obj['y']},{neighbor_obj['x']}). Given this context and logic, provide a spacial relationship between the two objects. Give 1 sentence and nothing else, just say the fact"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        edges[i][neighbor_id] = response.text

print(edges)


