
from urllib.request import urlopen
from bs4 import BeautifulSoup
import sys
import re
import time
import json
import configparser

def load_config():
    config = configparser.ConfigParser()
    config.read('./config.ini')
    
    return config

def save_config(config):
    f = open('./config.ini', 'w')
    config.write(f)


def main(index):
    url = 'https://www.insight.co.kr/news/'
    url = url + str(index)

    html = urlopen(url)
    bsObject = BeautifulSoup(html, 'html.parser')

    if bsObject.title is None:
        return

    head = bsObject.head
    summarization = head.find('meta', {'name':'description'})
    summarization = summarization.attrs['content']

    classification = head.find('meta', {'name': 'classification'})
    classification = classification.attrs['content']

    keywords = head.find('meta', {'name': 'keywords'})
    keywords = keywords.attrs['content']

    section = head.find('meta', {'property': 'article:section'})
    section = section.attrs['content']

    title = bsObject.title.get_text() 

    content = bsObject.find('div', {'class':'news-article-memo'})
    content = str(content)
#   content = content.get_text()

    date = bsObject.find('em', {'class': 'news-byline-date-edit'})

    if date is None:
        date = bsObject.find('em', {'class': 'news-byline-date-send'})
        date = date.get_text()
    else:
        date = date.get_text()

    return {'index': index,
            'title': title,
            'date': date,
            'class': classification,
            'section': section,
            'keywords': keywords,
            'summary': summarization,
            'content': content}

if __name__=='__main__':
    print('[Crawling Start]')
    config = load_config()

    index = int(config['INDEX']['CurrentIndex'])
    output = config['OUTPUT']['Path']
    output = output + 'insight_crawling_' + time.strftime('%Y%m%d', time.localtime(time.time())) + '.json'

    print('Start: {}'.format(index))
    print('Path: {}'.format(output))

    result = []
    patient = 0
    output_file = open(output, 'a')
    count = 0

    while(patient < 100):
        try:
            dict_result = main(index)
            index += 1
            count += 1
        except AttributeError as e:
            patient += 1
            index += 1
            continue

        if dict_result:
            result.append(dict_result)
            patient = 0
        else:
            patient += 1

        if index % 100 == 0:
            print('index: {}'.format(index))
            json_data = json.dumps(result, indent=4, ensure_ascii=False)
            output_file.write(json_data)
            result = []

    print(result)

    if len(result) > 0:
        json_data = json.dumps(result, indent=4, ensure_ascii=False)
        output_file.write(json_data)
        result = []

    print('Crawling Completed {} News.'.format(count))
    print('Last Index: {}'.format(index))

    f = open('./config.ini', 'w')
    config['INDEX']['CurrentIndex'] = str(index - patient)
    
    config.write(f)

