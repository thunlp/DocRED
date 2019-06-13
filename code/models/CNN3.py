import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class CNN3(nn.Module):
	def __init__(self, config):
		super(CNN3, self).__init__()
		self.config = config
		self.word_emb = nn.Embedding(config.data_word_vec.shape[0], config.data_word_vec.shape[1])
		self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
		self.word_emb.weight.requires_grad = False


		# self.char_emb = nn.Embedding(config.data_char_vec.shape[0], config.data_char_vec.shape[1])
		# self.char_emb.weight.data.copy_(torch.from_numpy(config.data_char_vec))
		# char_dim = config.data_char_vec.shape[1]
		# char_hidden = 100
		# self.char_cnn = nn.Conv1d(char_dim,  char_hidden, 5)

		self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
		self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

		input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size #+ char_hidden

		self.out_channels = 200
		self.in_channels = input_size

		self.kernel_size = 3
		self.stride = 1
		self.padding = int((self.kernel_size - 1) / 2)

		self.cnn_1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
		self.cnn_2 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
		self.cnn_3 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
		self.max_pooling = nn.MaxPool1d(self.kernel_size, stride=self.stride, padding=self.padding)
		self.relu = nn.ReLU()

		self.dropout = nn.Dropout(config.cnn_drop_prob)

		self.bili = torch.nn.Bilinear(self.out_channels+config.dis_size, self.out_channels+config.dis_size, config.relation_num)
		self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)


	def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h):
		# para_size, char_size, bsz = context_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)
		# context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
		# context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)

		sent = torch.cat([self.word_emb(context_idxs), self.coref_embed(pos), self.ner_emb(context_ner)], dim=-1)

		sent = sent.permute(0, 2, 1)

		# batch * embedding_size * max_len
		x = self.cnn_1(sent)
		x = self.max_pooling(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.cnn_2(x)
		x = self.max_pooling(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.cnn_3(x)
		x = self.max_pooling(x)
		x = self.relu(x)
		x = self.dropout(x)

		context_output = x.permute(0, 2, 1)
		start_re_output = torch.matmul(h_mapping, context_output)
		end_re_output = torch.matmul(t_mapping, context_output)

		s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
		t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)

		predict_re = self.bili(s_rep, t_rep)

		return predict_re
