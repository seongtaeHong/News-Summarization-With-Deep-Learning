#-*-coding:utf-8-*-
# Created by Microsoft Corporation
# Licensed under the MIT license.

# Modified by Adaptive Intelligence Research Lab(https://air.changwon.ac.kr/)., 2020. 01. ~

import os
import json
import logging
import torch
import sys
import re
import numpy as np
from torch.utils import data
from time import time

def cleaning(sentence):

	sent = sentence.strip()

# 	sent = re.sub('\[[^\]]*\]','',sent)
# 	sent = re.sub('\([^\)]*\)','',sent)
# 	sent = re.sub('[^ㅏ-ㅣㄱ-ㅎ가-힣0-9a-zA-Z\.%, ]',' ', sent)
	sent = re.sub('  *',' ',sent).strip()

	return sent

class Dataset(data.Dataset):
	def __init__(self, filepath, tokenizer, task='pretrain', corruption_ratio=0.15, span_length=3, max_source_length=None, max_target_length=None):
		
		self.filepath = filepath
		self.task = task
		self.tokenizer = tokenizer
		self.corruption_ratio = corruption_ratio
		self.span_length = span_length

		self.real_item_size = 0

		if max_source_length is None: self.max_source_length = 10e+10
		else: self.max_source_length = max_source_length
		if max_target_length is None: self.max_target_length = 10e+10
		else: self.max_target_length = max_target_length

		if os.path.isdir(filepath):
			logging.info('training data is DIR.')
			self.filelist = sum([[os.path.join(d[0],f) for f in d[-1]] for d in list(os.walk(filepath))],[])
			self.filelist = sorted(self.filelist)
			self.filelist = self.filelist[1:]
			self.len_filelist = len(self.filelist)

			self.source, self.source_length, self.target, self.target_length, data_max_source, data_max_target = self.load_data(self.filelist.pop(0))
			self.item_size = int(self.len_filelist * len(self.source)) 
		else:
			self.source, self.source_length, self.target, self.target_length, data_max_source, data_max_target = self.load_data(filepath)
			self.item_size = len(self.source)
		
		if max_source_length is None: self.max_source_length = data_max_source
		if max_target_length is None: self.max_target_length = data_max_target

		logging.info('Total batch : {}'.format(self.item_size))
		logging.info('Max Source Length : {}'.format(self.max_source_length))
		logging.info('Max Target Length : {}'.format(self.max_target_length))

	def load_data(self, filename):
		logging.info('Load {} '.format(filename))
		
		sources = list()
		source_lengths = list()
		targets = list()
		target_lengths = list()

		with open(filename, 'r') as ifp:
			key = os.path.splitext(filename)[-1]
			if key in ['.jsonl']:
				data = [json.loads(d) for d in ifp]
				data = [[d['source'],d['target']] if self.task=='finetune' else d['source'] for d in data]
			elif key in ['json']:
				data = json.load(d)
				data = [[d['source'],d['target']] if self.task=='finetune' else d['source'] for d in data]
			elif key in ['.tsv','.txt']:
				data = [d for d in ifp]
				data = [d.strip().split('\t') if self.task=='finetune' else d.strip().split('\t')[0] for d in data]
			else:
				raise KeyError('No rule for {} filetype.'.format(key))

		for index, item in enumerate(data):
			start_time = time()
			if self.task == 'pretrain':
				source = item
				if type(source) == list: source = source[0]
				soruce = cleaning(source) 
				
				source_length = min(len(self.tokenizer.encode(source)), self.max_source_length)
				if source_length < 5: continue

				sources.append(source)
				source_lengths.append(source_length)
			else:
				if len(item) < 2:
					continue
				else:
					source, target = item
				if type(source) == list: source = source[0]
				if type(target) == list: target = target[0]
				source = cleaning(source)
				target = cleaning(target)

				source_length = len(self.tokenizer.encode(source))
				target_length = len(self.tokenizer.encode(target))

				sources.append(source)
				source_lengths.append(source_length)
				targets.append(target)
				target_lengths.append(target_length)
			sys.stderr.write('\r{}-{:.2f}'.format(index, time()-start_time))
			sys.stderr.flush()
					
		logging.info('- size: {}'.format(len(sources)))
		
		return sources, source_lengths, targets, target_lengths, max(source_lengths+[0]), max(target_lengths+[0])
		
	
	def __getitem__(self, index):
		if len(self.source) == 0:
			if os.path.isdir(self.filepath):
				if len(self.filelist) == 0:
					self.filelist = sum([[os.path.join(d[0],f) for f in d[-1]] for d in list(os.walk(self.filepath))],[])
					self.len_filelist = len(self.filelist)
					self.item_size = self.real_item_size
					self.real_item_size = 0
				self.source, self.source_length, self.target, self.target_length, _, _ = self.load_data(self.filelist.pop(0))
			else:
				self.source, self.source_length, self.target, self.target_length, _, _ = self.load_data(self.filepath)

		self.real_item_size += 1
		if self.task=='pretrain':
			return self.source.pop(0), self.source_length.pop(0), None, None
		else:
			return self.source.pop(0), self.source_length.pop(0), self.target.pop(0), self.target_length.pop(0)

	def __len__(self):
		## for setting total_batch, 10%: removed data if source_length < 5
		#return int(1000000 * len(self.filelist) * 0.9)
		return self.item_size

def collate_fn(data, tokenizer, corruption_ratio, span_length, max_source_length, max_target_length, task='pretrain'):

	sources, source_lengths, targets, target_lengths = zip(*data)

	sources = []
	targets_in = []
	targets_out = []
	target_lengths = []

	all_special_ids = set(tokenizer.all_special_ids)
	special_ids = set(tokenizer.additional_special_tokens_ids)

	for items in data:
		if task=='pretrain':
			source, source_length, _, _ = items
			
			masked_source_word = tokenizer.encode(source)
			masked_source_word = masked_source_word[:max_source_length]
			source_length = len(masked_source_word)
			masked_word = dict()
			num_mask = max(int(source_length*corruption_ratio/span_length),1)
			mask_count = 1

			patient = 0
			while(num_mask > mask_count-1):
				if patient > 30:
					masked_word = dict()
					break
				mask_index = torch.randint(source_length, size=(1,)).item()
				temp_word = masked_source_word[mask_index:mask_index+span_length]
				
				if all_special_ids.intersection(set(temp_word)) != set():
					patient += 1
					continue
				key = '<extra_id_{}>'.format(mask_count)
				key = tokenizer._convert_token_to_id(key)
				masked_word[key] = temp_word

				masked_source_word[mask_index:mask_index+span_length] = [key]
				mask_count += 1
			if masked_word == dict(): continue
			masked_word = [[d]+masked_word[d] for d in masked_source_word if d in special_ids and d in masked_word]
			masked_word = sum(masked_word, [])

			tokenized_source_index = masked_source_word[:max_source_length] + [tokenizer.pad_token_id]*(max_source_length-len(masked_source_word))

			masked_word = masked_word[:max_source_length-1]
			tokenized_target_in_index = [tokenizer.bos_token_id] + masked_word
			tokenized_target_in_index = tokenized_target_in_index + [tokenizer.pad_token_id]*(max_source_length-len(tokenized_target_in_index))

			tokenized_target_out_index = masked_word + [tokenizer.eos_token_id]
			tokenized_target_out_index = tokenized_target_out_index + [tokenizer.pad_token_id]*(max_source_length-len(tokenized_target_out_index))

			target_length = tokenized_target_out_index.index(tokenizer.pad_token_id) if tokenizer.pad_token_id in tokenized_target_out_index else len(tokenized_target_out_index)

		else:
			source, source_length, target, target_length = items

			# source
# 			tokenized_source = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(source), add_special_tokens=True)[:max_source_length]
			tokenized_source = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(source))[:max_source_length]
			tokenized_source_index = tokenized_source + [tokenizer.pad_token_id]*(max_source_length-len(tokenized_source))

			tokenized_target = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target))[:max_target_length-1]
			tokenized_target_in_index = [tokenizer.pad_token_id]+tokenized_target
			tokenized_target_in_index = tokenized_target_in_index+ [tokenizer.pad_token_id]*(max_target_length - len(tokenized_target_in_index))

			tokenized_target_out_index = tokenized_target + [tokenizer.eos_token_id]
			tokenized_target_out_index = tokenized_target_out_index + [tokenizer.pad_token_id]*(max_target_length - len(tokenized_target_out_index))

			target_length = tokenized_target_out_index.index(tokenizer.pad_token_id) if tokenizer.pad_token_id in tokenized_target_out_index else len(tokenized_target_out_index)

		sources.append(tokenized_source_index)
		targets_in.append(tokenized_target_in_index)
		targets_out.append(tokenized_target_out_index)
		target_lengths.append(target_length)

	source_tensor = torch.tensor(sources)
	target_in_tensor = torch.tensor(targets_in)
	target_out_tensor = torch.tensor(targets_out)
	target_lengths = torch.tensor(target_lengths)
	return source_tensor, target_in_tensor, target_out_tensor, target_lengths


def get_dataloader(filepath, tokenizer, batch_size, task='pretrain', corruption_ratio=0.15, span_length=2, max_source_length=None, max_target_length=None, shuffle=True):
	logging.info('Reading from {}.'.format(filepath))

	dataset = Dataset(filepath, tokenizer, task=task, corruption_ratio=corruption_ratio,span_length=span_length, max_source_length=max_source_length, max_target_length=max_target_length)

	if max_source_length is None: max_source_length = dataset.max_source_length
	if max_target_length is None: max_target_length = dataset.max_target_length

	data_loader = torch.utils.data.DataLoader(dataset,
			                                  batch_size=batch_size, 
											  shuffle=shuffle, 
											  num_workers=0,
											  collate_fn=lambda data: collate_fn(data, tokenizer, corruption_ratio, span_length, max_source_length, max_target_length, task=task))
	return data_loader

def parse_args():
	from argparse import ArgumentParser
	parser = ArgumentParser(description='M3 Trainer')
	# training
	parser.add_argument('-t', '--task', required=True, choices=['pretrain', 'finetune'])
	# hyper params
	parser.add_argument('-b', '--batch', default=16, type=int)
	parser.add_argument('-mr', '--mask_ratio', default=0.15, type=float, help='ratio of making token in source')
	parser.add_argument('-ml', '--mask_length', default=3, type=int, help='masked word length')
	# dataset
	parser.add_argument('-td', '--dataset', required=True, type=str)
	parser.add_argument('-ms', '--max_source_length', default=512, type=int)
	parser.add_argument('-mt', '--max_target_length', default=None, type=int)
	# etc
	parser.add_argument('-tk','--tokenizer_path', default='tokenizer/tokenizer_30000_addUmjeol/spiece.model', help='path of pretrained tokenizer model file')
	args = parser.parse_args()

	return args


if __name__=='__main__':
	from transformers import M3Tokenizer

	args = parse_args()

	tokenizer = M3Tokenizer.from_pretrained(args.tokenizer_path)

	train_dataloader = get_dataloader(args.dataset, tokenizer, args.batch,
			                          task=args.task,
									  corruption_ratio=args.mask_ratio,
									  span_length=args.mask_length,
									  max_source_length = args.max_source_length,
									  max_target_length = args.max_target_length,
									  shuffle = False)
	count = 0
	while(1):
		for di, d in enumerate(train_dataloader):
			start = time()
			sys.stderr.write('\rCHECK UTILS: epoch {}\tstep {}\ttime {:.5f}'.format(count,di,time()-start))
			count += 1
