from google import genai
import json
import numpy as np

def generate_edge_relationships(context, use_llm):
    client = genai.Client(api_key="AIzaSyB9mTbKNa5SrHEvLK4HzdACiX2-VM4iNF0")

    edges = [["" for i in range(len(context))] for j in range(len(context))]
    context_matrix = [["" for _ in range(len(context))] for _ in range(len(context))]

    for obj in context:
        obj['x'] = round(obj['x'], 2)
        obj['y'] = round(obj['y'], 2)
        obj['z'] = round(obj['z'], 2)

    for i, obj in enumerate(context):
        for neighbor_id in obj['neighbors']:
            neighbor_obj = context[neighbor_id]
            
            prompt = f"context: {obj['name']} is at position (x,y,z) = ({obj['x']},{obj['y']},{obj['x']}). The object has a {neighbor_obj['name']} at position (x,y,z) = ({neighbor_obj['x']},{neighbor_obj['y']},{neighbor_obj['x']}). Given this context and logic, provide a spacial relationship between the two objects. Give 1 sentence and nothing else, just say the fact"
            context_matrix[i][neighbor_id] = f"1. object has {obj['name']} position (x,y,z) = ({obj['x']},{obj['y']},{obj['x']}). Second object {neighbor_obj['name']} has position (x,y,z) = ({neighbor_obj['x']},{neighbor_obj['y']},{neighbor_obj['x']})."
            if use_llm:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                edges[i][neighbor_id] = response.text
    
            else:
                edges[i][neighbor_id] = "USE_LLM=False"
    
    return context_matrix, edges

if __name__ == '__main__':
    with open('/teamspace/studios/this_studio/3dscenegraph/output/apple_banana_scan/result/objects.json', 'r') as file:
        context = json.load(file)

    context_matrix, edges = generate_edge_relationships(context)
    print(edges)




