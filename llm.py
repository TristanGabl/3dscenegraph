from ollama import chat
import json

def llm_response(message: str) -> str:
    response = chat(model='gemma:2b', messages=[
        {
            'role': 'user',
            'content': message,
        },
    ])
    return response['message']['content']

if __name__ == '__main__':
    with open('/teamspace/studios/this_studio/3dscenegraph/output/apple_banana_scan/result/objects.json', 'r') as file:
        context = json.load(file)
    
    

    print(llm_response('Use the context in the following and provide a  relationship, for example "(Object1)->on top of->(Object2)", do not analyze or explain or say anything else:: Apple is neighbor to banana, banana is neighbor to apple, banana is neighbor to table, apple is neighbor to table, table is neighbor to banana, table is neighbor to apple'))
    print(llm_response('Use the following to create a logical relationship between the two objects, for example "(Object1)->on top of->(Object2)", do not analyze or explain or say anything else:: Apple is geometrically next to table'))
    print(llm_response('Use the following context to deduce a spacial relationship between the two objects, for example "(Object1)->*relationship*->(Object2)", do not analyze or explain or say anything else, coordinates are (x,y,z): Apple at position (0,0,0); Banana at position (0,0.1,0)'))

