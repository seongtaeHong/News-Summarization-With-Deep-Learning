
import json
import sys

open_file = sys.argv[1]

json_data = json.load(open(open_file, 'r'))
result = []

for data in json_data:
    content = data['content']
    summary = data['summary']
    
    new_line = content.strip() + '\t' + summary.strip()
    result.append(new_line)

output_file = open(sys.argv[2], 'w')
output_file.write('\n'.join(result))
