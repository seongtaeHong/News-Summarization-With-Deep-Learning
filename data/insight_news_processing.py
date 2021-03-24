
import json
import re
import argparse

# Argument 설정(-i: 입력 파일(.json) 경로, -o: 출력 파일명)
def argument_parsing():
	parser = argparse.ArgumentParser(description='JSON 형식의 파일을 TSV 파일로 변경해주는 파이썬 프로그램')
	parser.add_argument('-i', '--input_path', required=True, type=str, help='입력 파일 경로(.json)')
	parser.add_argument('-o', '--output_path', default='./output.tsv', type=str, help='출력 파일명')

	args = parser.parse_args()
	return args

# 뉴스에서 첫 줄과 맨 끝 줄을 지우는 메소드
# 인사이트 기사에서 첫 줄은 기자의 생각이 담겨있는데 그대로 학습시키면 아마 첫 줄만 가져올 것
# 마지막 줄은 기사의 출처나 기자 이름이 적혀있어서 필요없는 데이터
def processing_content(content):
	if '다.' not in content:
		return ''
	content = content[content.index('다.')+len('다.'):]

	if '다.' not in content:
		return ''

	content = content[:content.rindex('다.') + len('다.')]
	content = remove_special_symbol(content)

	if content[0] == ' ' and len(content) > 1: content = content[1:]

	return content

# 크롤링으로 가져온 데이터에 불필요한 텍스트가 포함되어있음
# TSV 파일로 변환하기 위해 \t를 삭제하고 한 줄에 한 데이터가 입력되게 \n을 삭제
def remove_special_symbol(content):
		content = content.replace('\r\n', ' ')
		content = content.replace('\n', ' ')
		content = content.replace('\t', ' ')
		content = content.replace('\xa0', ' ')
		content = content.replace('  ', ' ')

		return content

# 첫 줄이 사진에 대한 설명이면 기자 이름이 완전히 삭제되지 않는 경우 발생
# 그에 대한 예외처리 메소드
def remove_reporter(content):

	# 첫 줄에 '=' 가 포함되면 "*** 기자 = 어쩌구저쩌구이다." 하는 내용
	if '=' in content.split('다.')[0]:
		content = content[content.index('='):]
		
		if '다.' in content:
			content = content[content.index('다.') + len('다.'):]

	return content

def main(data_path, output_path):

	with open(data_path, 'r') as f:
		json_data = json.load(f)

	lines = []

	# 중간 Logging도 실시
	for index, data in enumerate(json_data):
		print('\nbefore: ', data['content'])
		content = processing_content(data['content'])
		content = remove_reporter(content)
		content = content.strip()
		print('\nafter: ', content)

		if content == '':
			print('content missed: {}'.format(data['index']))
			continue
		elif len(content) < 30:
			print('content is too short: ', content)

		summary = remove_special_symbol(data['summary'])
		line = content + '\t' + summary

		lines.append(line)

	f = open(output_path, 'w')
	f.write('\n'.join(lines))


if __name__=='__main__':
	args = argument_parsing()
	data_path = args.input_path
	output_path = args.output_path

	main(data_path, output_path)

