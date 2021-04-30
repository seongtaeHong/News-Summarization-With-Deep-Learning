import sys, torch, logging, os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import nltk
smooth_func = nltk.translate.bleu_score.SmoothingFunction()

from transformers import *
import transformers
from utils import *
from masked_cross_entropy import MaskedCrossEntropyLoss

from time import time

import warnings

torch.manual_seed(100)    ## seed 고정

warnings.filterwarnings(action='ignore')

logging.getLogger().setLevel(logging.INFO)

def parse_args():
	parser = ArgumentParser(description='M3 Trainer')
	# training
	parser.add_argument('-t', '--task', default='finetune', type=str)
	parser.add_argument('--pgn', action='store_true', help='using pgn')
	# hyper params
	parser.add_argument('-b', '--batch', default=16, type=int)
	parser.add_argument('-mr', '--mask_ratio', default=0.15, type=float, help='ratio of making token in source')
	parser.add_argument('-ml', '--mask_length', default=3, type=int, help='masked word length')
	# dataset
	parser.add_argument('-td', '--test_dataset', required=True, type=str)
	parser.add_argument('-ms', '--max_source_length', default=512, type=int)
	parser.add_argument('-mt', '--max_target_length', default=None, type=int)
	# etc
	parser.add_argument('-tk','--tokenizer_path', default='tokenizer/sentencepiece.model', help='path of pretrained tokenizer model file')
	parser.add_argument('-w','--weights', default=None, help='path of pretrained model weight file')
	parser.add_argument('-wl', '--weight_lists', default=None, help='path of pretrained model weight directory')
	parser.add_argument('-s','--save_path',default='prediction.txt', type=str, help='path of prediction result file')
	parser.add_argument('-l','--loop', default=0, type=int, help='decoder repetition count')
	parser.add_argument('-f','--format', default='word', type=str, help='output format, [\'word\',\'token\']')
	args = parser.parse_args()

	return args

def save_model(path):
	os.makedirs(path, exist_ok=True)
	if torch.cuda.device_count() > 1:
		model.module.save_pretrained(path)
	else:
		model.save_pretrained(path)

def init_model(args):
	# Load dataset, tokenizer, model from pretrained model/vocabulary
	
	## google의 sentencepiece tokenizer 
	tokenizer = transformers.T5Tokenizer.from_pretrained(args.tokenizer_path)
	special_tokens = ['<mask{}>'.format(d) for d in range(0,100)]
	special_tokens += ['<unused{}>'.format(d) for d in range(0,100)]
	special_tokens_dict = {
			'bos_token':'<s>',
			'sep_token':'<sep>',
			'cls_token':'<cls>',
			'mask_token':'<mask>',
			'additional_special_tokens': special_tokens
			}
	tokenizer.add_special_tokens(special_tokens_dict)
	if args.weights == None:
		model = transformers.T5ForConditionalGeneration(vocab_size=tokenizer.vocab_size)
	else:
		logging.info('Load {}.'.format(args.weights))
		model = transformers.T5ForConditionalGeneration.from_pretrained(args.weights)
		if model.config.vocab_size != tokenizer.vocab_size:
			logging.info('Resize embedding {} -> {}.'.format(model.config.vocab_size, tokenizer.vocab_size))
			model.resize_token_embeddings(tokenizer.vocab_size)
	model.eval()

	loss_func = MaskedCrossEntropyLoss()
	if torch.cuda.device_count() > 1:
		logging.info('Training in multi GPU mode using {} GPUs.'.format(torch.cuda.device_count()))
		model = torch.nn.DataParallel(model)
	if torch.cuda.is_available():
		model.to('cuda')
		loss_func.to('cuda')
	return tokenizer, model, loss_func

def get_output(args, tokenizer, model):
	
	if torch.cuda.is_available(): batch_size = args.batch*torch.cuda.device_count()
	else: batch_size = args.batch
	dataloader = get_dataloader(args.test_dataset, tokenizer, batch_size,
			task=args.task,
			max_source_length=args.max_source_length,
			max_target_length=args.max_target_length)

	string = list()
	token = list()
	tqdm_loader = tqdm(dataloader)

	f = open('{}/{}'.format(args.weights, args.save_path),'w') 
	for b_index, batch in enumerate(tqdm_loader):
		source, target_in, target_out, target_length = batch

		if torch.cuda.is_available():
			source = source.cuda()
			target_in = target_in.cuda()
			target_out = target_out.cuda()

		#predict = model.generate(input_ids=source)
		outputs = model(input_ids=source, labels=target_out)
		predict = outputs.logits 
# 		if type(logit) == list:
# 			predict = torch.cat(logit).detach()
# 		else: predict = logit.detach()

		predict = torch.max(predict, dim=2)[1]

		for pidx in range(predict.shape[0]):
			pred_out = tokenizer.convert_ids_to_tokens(predict[pidx])
			gold_out = tokenizer.convert_ids_to_tokens(target_out[pidx])
			input_out = tokenizer.convert_ids_to_tokens(source[pidx])

			pred_out = pred_out[:pred_out.index(tokenizer.eos_token) if tokenizer.eos_token in pred_out else len(pred_out)]
			gold_out = gold_out[:gold_out.index(tokenizer.eos_token) if tokenizer.eos_token in gold_out else len(gold_out)]
			input_out = input_out[:input_out.index(tokenizer.pad_token) if tokenizer.pad_token in input_out else len(input_out)]

			if args.format == 'word':
				pred_str = ''.join(pred_out).replace(tokenizer.bos_token,'').replace('▁',' ').strip()
				gold_str = ''.join(gold_out).replace(tokenizer.bos_token,'').replace('▁',' ').strip()
				input_str = ''.join(input_out).replace(tokenizer.bos_token,'').replace('▁',' ').strip()
			elif args.format == 'token':
				pred_str = ' '.join(pred_out).replace(tokenizer.bos_token,'').strip()
				gold_str = ' '.join(gold_out).replace(tokenizer.bos_token,'').strip()
				input_str = ' '.join(input_out).replace(tokenizer.bos_token,'').strip()
			else:
				raise KeyError('ERROR: "{}" is not format type'.format(args.format))

			pred_str = pred_str.replace('<pad> ', '')
			pred_str = pred_str.replace('<pad>', '')

			f.write(f'SRCE:{input_str}\nGOLD:{gold_str}\nPRED:{pred_str}\n\n')
	f.close()

if __name__ == '__main__':

	## Example
	args = parse_args()
	
	if args.weights is None and args.weight_lists is None:
		print('needed weight directories')
		sys.exit()
	
	if args.weights is None:
		directory = args.weight_lists
		weight_lists = os.listdir(directory)
		weight_lists.sort()

		for weight in weight_lists:
			if 'log' in weight:
				continue
			args.weights = directory + weight + '/'
			logging.info('weight: {}'.format(args.weights))
			tokenizer, model, loss_func = init_model(args)
			get_output(args, tokenizer, model)
	else:
		tokenizer, model, loss_func = init_model(args)
		
		get_output(args, tokenizer, model)

	sys.exit()

#	assert args.weights is not None

	tokenizer, model, loss_func = init_model(args)

	get_output(args, tokenizer, model)
