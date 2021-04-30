import sys
import torch
from torch.nn import Module
from torch.autograd import Variable
from torch.nn import functional


class _Loss(Module):
	def __init__(self, size_average=None, reduce=None, reduction='mean'):
		super(_Loss, self).__init__()
		if size_average is not None or reduce is not None:
			self.reduction = _Reduction.legacy_get_string(size_average, reduce)
		else:
			self.reduction = reduction


class _WeightedLoss(_Loss):
	def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
		super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
		self.register_buffer('weight', weight)

class MaskedCrossEntropyLoss(_WeightedLoss):

	__constants__ = ['weight', 'ignore_index', 'reduction']

	def __init__(self, weight=None, size_average=None, ignore_index=-100,
				 reduce=None, reduction='mean'):
		super(MaskedCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
		self.ignore_index = ignore_index

	def sequence_mask(self, sequence_length, max_len=None):
		if max_len is None:
			max_len = sequence_length.data.max()
		batch_size = sequence_length.size(0)
		seq_range = torch.arange(0, max_len).long()
		seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
		seq_range_expand = Variable(seq_range_expand)
		if sequence_length.is_cuda:
			seq_range_expand = seq_range_expand.cuda()
		seq_length_expand = (sequence_length.unsqueeze(1)
							 .expand_as(seq_range_expand))
		return seq_range_expand < seq_length_expand

	def forward(self, logits, target, length):
# 		logits, target, length = inputs
		
		## length type check
		if length.dtype != torch.LongTensor:
			length = length.type(torch.LongTensor)
		## XXX segmentation fault (core dump) 발생, 그냥 두면 서버도 먹통이 되니 빠르게 재부팅할 것
# 		if torch.cuda.is_available():
# 			length = Variable(torch.LongTensor(length)).cuda()
# 		else:
# 			length = Variable(torch.LongTensor(length))	
# 		print('DONE length to variable')

		# logits_flat: (batch * max_len, num_classes)
		logits_flat = logits.view(-1, logits.size(-1)) ## -1 means infered from other dimentions
		# log_probs_flat: (batch * max_len, num_classes)
		log_probs_flat = functional.log_softmax(logits_flat,dim=1)
		# target_flat: (batch * max_len, 1)
		target_flat = target.view(-1, 1)
		# losses_flat: (batch * max_len, 1)
		losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
		# losses: (batch, max_len)
		losses = losses_flat.view(*target.size())
		# mask: (batch, max_len)
		mask = self.sequence_mask(sequence_length=length, max_len=target.size(1))  
		if torch.cuda.is_available(): mask = mask.cuda()
		losses = losses * mask.float()
		loss = losses.sum() / length.float().sum()
		return loss
