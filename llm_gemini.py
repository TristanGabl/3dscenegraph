from google import genai
import json
import numpy as np

def generate_edge_relationships(context):
    client = genai.Client(api_key="AIzaSyB9mTbKNa5SrHEvLK4HzdACiX2-VM4iNF0")

    edges = [["" for i in range(len(context))] for j in range(len(context))]
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
    
    return edges

if __name__ == '__main__':
    with open('/teamspace/studios/this_studio/3dscenegraph/output/apple_banana_scan/result/objects.json', 'r') as file:
        context = json.load(file)

    edges = generate_edge_relationships(context)
    print(edges)




