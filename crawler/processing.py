
import sys

open_path = sys.argv[1]
output_path = sys.argv[2]

f = open(open_path, 'r')

lines = f.readlines()
result = []

for line in lines:
    if line == '][\n':
        result[-1] = result[-1] + ','
        print('skipped: {}'.format(len(result)))
    else:
        result.append(line.strip())


output_file = open(output_path, 'w')

import json
from bs4 import BeautifulSoup

print('json, bs libraries imported')

raw_text = '\n'.join(result)

json_data = json.loads(raw_text)
result = []

for data in json_data:
    if data['section'] == '인사' or data['section'] == '엔터테인먼트':
        continue
    
    if '<p><br/></p>' not in data['content']:
        continue
    
    content = data['content']
    content = content.replace('\xa0', ' ')

    if '기자 =' not in content:
        print('기자 not in {}'.format(data['index']))
        continue

    content = content[content.index('기자 =') + len('기자 ='):]

    # <p><br/></p>
    if '<p><br/></p>' not in content:
        print('<p><br/></p> not in {}'.format(data['index']))
        continue

    content = content[content.index('<p><br/></p>') + len('<p><br/></p>'):]

    content = content.split('<p><br/></p>')
    if '기자' in content[-1]:
        content = content[:-1]

    line_content = []

    for c in content:
        if 'class' in c:
            print('line skipped that has class word: {}.'.format(data['index']))
            continue
        
        line_content.append(c)

    content = '<p><br/></p> '.join(line_content)

    content = BeautifulSoup(content, 'html.parser').get_text()
    
    if content.replace(' ', '') == '':
        print('content missed: {}'.format(data['index']))
        continue

    if data['summary'] in content:
        print('content has summary: {}'.format(data['index']))
        content.replace(data['summary'], '')

    content = content.replace('\r\n', ' ')
    content = content.replace('\n', ' ')
    content = content.replace('\t', ' ')
    content = content.replace('  ', ' ')

    data['content'] = content

    result.append(data)
    
output_file.write(json.dumps(result, indent=4, ensure_ascii=False))


