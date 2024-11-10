import json

with open('numbercompress.json') as json_file:
    data = json.load(json_file)
    print(type(data))
    print("layer_type" in data)
    print("layer_number" in data)