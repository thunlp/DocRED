# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch.nn.functional as F


IGNORE_INDEX = -100
TRAIN_LIMIT = 3600
test_evidence = False

class Accuracy(object):
	def __init__(self):
		self.correct = 0
		self.total = 0
	def add(self, is_correct):
		self.total += 1
		if is_correct:
			self.correct += 1
	def get(self):
		if self.total == 0:
			return 0.0
		else:
			return float(self.correct) / self.total
	def clear(self):
		self.correct = 0
		self.total = 0 

class EviConfig(object):
	def __init__(self, args):
		self.acc_NA = Accuracy()
		self.acc_not_NA = Accuracy()
		self.acc_total = Accuracy()
		self.data_path = './prepro_data'
		self.use_bag = False
		self.use_gpu = True
		self.is_training = True
		self.max_length = 512
		self.pos_num = 2 * self.max_length
		self.entity_num = self.max_length
		self.relation_num = 97
		self.coref_size = 20
		self.entity_type_size = 20
		self.max_epoch = 20
		self.opt_method = 'Adam'
		self.optimizer = None
		self.drop_prob = 0.5 # for cnn
		self.keep_prob = 0.8 # for lstm
		self.checkpoint_dir = './checkpoint'
		self.test_result_dir = './test_result'
		self.test_epoch = 5
		self.pretrain_model = None


		self.word_size = 100
		self.epoch_range = None
		self.dropout = 0.5
		self.period = 50

		self.ins_batch_size = 40
		self.test_ins_batch_size = self.ins_batch_size
		self.batch_size = 4000


		self.char_limit = 16
		self.sent_limit = 25
		self.dis2idx = np.zeros((512), dtype='int64')
		self.dis2idx[1] = 1
		self.dis2idx[2:] = 2
		self.dis2idx[4:] = 3
		self.dis2idx[8:] = 4
		self.dis2idx[16:] = 5
		self.dis2idx[32:] = 6
		self.dis2idx[64:] = 7
		self.dis2idx[128:] = 8
		self.dis2idx[256:] = 9
		self.dis_size = 20

		self.train_prefix = args.train_prefix
		self.test_prefix = args.test_prefix
		self.output_file = args.output_file


	def set_data_path(self, data_path):
		self.data_path = data_path
	def set_max_length(self, max_length):
		self.max_length = max_length
		self.pos_num = 2 * self.max_length
	def set_num_classes(self, num_classes):
		self.num_classes = num_classes
	def set_window_size(self, window_size):
		self.window_size = window_size
	def set_pos_size(self, pos_size):
		self.pos_size = pos_size
	def set_word_size(self, word_size):
		self.word_size = word_size
	def set_max_epoch(self, max_epoch):
		self.max_epoch = max_epoch
	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
	def set_opt_method(self, opt_method):
		self.opt_method = opt_method
	def set_drop_prob(self, drop_prob):
		self.drop_prob = drop_prob
	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir
	def set_test_epoch(self, test_epoch):
		self.test_epoch = test_epoch
	def set_pretrain_model(self, pretrain_model):
		self.pretrain_model = pretrain_model
	def set_is_training(self, is_training):
		self.is_training = is_training
	def set_use_bag(self, use_bag):
		self.use_bag = use_bag
	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu
	def set_epoch_range(self, epoch_range):
		self.epoch_range = epoch_range
	
	def load_train_data(self):
		print("Reading training data...")

		prefix = 'dev_train'
		self.data_train_word = np.load(os.path.join(self.data_path, prefix+'_word.npy'))
		self.data_train_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
		self.data_train_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy'))
		self.data_train_char = np.load(os.path.join(self.data_path, prefix+'_char.npy'))
		self.train_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

		print("Finish reading")

		self.train_len = ins_num = self.data_train_word.shape[0]
		assert(self.train_len==len(self.train_file))

		self.train_order = list(range(ins_num))
		self.train_batches = ins_num // self.ins_batch_size
		if ins_num % self.ins_batch_size != 0:
			self.train_batches += 1

	def load_test_data(self):
		print("Reading testing data...")

		self.data_char_vec = np.load(os.path.join(self.data_path, 'char_vec.npy'))
		self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
		self.id2rel = {v: k for k,v in self.rel2id.items()}

		prefix = self.test_prefix
		print (prefix)
		self.data_test_word = np.load(os.path.join(self.data_path, prefix+'_word.npy'))
		self.data_test_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
		self.data_test_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy'))
		self.data_test_char = np.load(os.path.join(self.data_path, prefix+'_char.npy'))
		self.test_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

		self.test_len = self.data_test_word.shape[0]
		assert(self.test_len==len(self.test_file))


		self.test_index = json.load(open(prefix+"_index.json"))


		self.total_evidence_recall = 0
		for ins in self.test_file:
			for label in ins['labels']:
				evidence = [int(e) for e in label['evidence']]
				self.total_evidence_recall += len(evidence)

		print ("total_evidence_recall:", self.total_evidence_recall)
		print ("Finish reading")

		self.test_batches = self.data_test_word.shape[0] // self.test_ins_batch_size
		if self.data_test_word.shape[0] % self.test_ins_batch_size != 0:
			self.test_batches += 1


		cur_batch = list(range(self.test_len))
		cur_batch.sort(key=lambda x: len(self.test_file[x]['vertexSet']))
		i = 0
		j = self.test_len-1
		# small vertexSet + big vertexSet as a pair
		self.test_order = []
		while i <= j:
			self.test_order.append(cur_batch[i])
			i += 1
			if i>j:
				break

			self.test_order.append(cur_batch[j])
			j -= 1

		assert(len(self.test_order)==self.test_len)


	def get_N2_train_batch(self):
		random.shuffle(self.train_order)

		context_idxs = torch.LongTensor(self.batch_size, self.max_length).cuda()
		context_pos = torch.LongTensor(self.batch_size, self.max_length).cuda()

		context_ner = torch.LongTensor(self.batch_size, self.max_length).cuda()
		context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit).cuda()

		relation_label = torch.LongTensor(self.batch_size).cuda()
		evidence_label = torch.Tensor(self.batch_size, self.sent_limit).cuda()
		sent_mask = torch.Tensor(self.batch_size, self.sent_limit).cuda()

		sent_h_mapping = torch.Tensor(self.batch_size, self.sent_limit, self.max_length).cuda()
		sent_t_mapping = torch.Tensor(self.batch_size, self.sent_limit, self.max_length).cuda()


		for b in range(self.train_batches):
			start_id = b * self.ins_batch_size
			cur_bsz = min(self.ins_batch_size, self.train_len - start_id)
			cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
			cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x]>0) , reverse = True)

			for mapping in [sent_h_mapping, sent_t_mapping, sent_mask, evidence_label, context_pos, relation_label]:
				mapping.zero_()

			max_sents = 0
			i = 0
			for w, index in enumerate(cur_batch):
				ins = self.train_file[index]
				Ls = ins['Ls']
				max_sents = max(max_sents, len(Ls) - 1)
				random.shuffle(ins['labels'])
				for label in ins['labels']:
					context_idxs[i].copy_(torch.from_numpy(self.data_train_word[index, :]))
					context_char_idxs[i].copy_(torch.from_numpy(self.data_train_char[index, :]))
					context_ner[i].copy_(torch.from_numpy(self.data_train_ner[index, :]))
					relation_label[i] = label['r']

					h_idx = label['h']
					t_idx = label['t']

					hlist = ins['vertexSet'][h_idx]
					tlist = ins['vertexSet'][t_idx]

					for h in hlist:
						context_pos[i, h['pos'][0]:h['pos'][1]] = 1

					for t in tlist:
						context_pos[i, t['pos'][0]:t['pos'][1]] = 2

					for e in label['evidence']:
						evidence_label[i, int(e)] = 1

					for j in range(len(Ls) - 1):
						sent_h_mapping[i, j, Ls[j]] = 1
						sent_t_mapping[i, j, Ls[j + 1] - 1] = 1
						sent_mask[i, j] = 1


					i += 1
					if i == self.batch_size:
						break
				if i == self.batch_size:
					break



			cur_bsz = i
			input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
			max_c_len = int(input_lengths.max())

			yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
				   'relation_label': relation_label[:cur_bsz].contiguous(),
				   'input_lengths' : input_lengths,
				   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
				   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'sent_h_mapping': sent_h_mapping[:cur_bsz, :max_sents, :max_c_len],
				   'sent_t_mapping': sent_t_mapping[:cur_bsz, :max_sents, :max_c_len],
				   'sent_mask': sent_mask[:cur_bsz, :max_sents],
				   'evidence_label': evidence_label[:cur_bsz, :max_sents]
				   }


	def get_real_test_batch(self):


		self.test_len  = len(self.test_index)

		self.test_order = list(range(self.test_len))
		self.test_batches = self.test_len // self.batch_size
		if self.test_len % self.batch_size != 0:
			self.test_batches += 1

		context_idxs = torch.LongTensor(self.batch_size, self.max_length).cuda()
		context_pos = torch.LongTensor(self.batch_size, self.max_length).cuda()

		context_ner = torch.LongTensor(self.batch_size, self.max_length).cuda()
		context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit).cuda()

		relation_label = torch.LongTensor(self.batch_size).cuda()

		sent_mask = torch.Tensor(self.batch_size, self.sent_limit).cuda()

		sent_h_mapping = torch.Tensor(self.batch_size, self.sent_limit, self.max_length).cuda()
		sent_t_mapping = torch.Tensor(self.batch_size, self.sent_limit, self.max_length).cuda()


		for b in range(self.test_batches):
			start_id = b * self.batch_size
			cur_bsz = min(self.batch_size, self.test_len - start_id)
			cur_batch = list(self.test_order[start_id : start_id + cur_bsz])

			cur_batch.sort(key=lambda x: np.sum(self.data_test_word[self.test_index[x]['index']]>0) , reverse = True)

			for mapping in [sent_h_mapping, sent_t_mapping, sent_mask, context_pos, relation_label]:
				mapping.zero_()

			max_sents = 0
			evidences = []
			sents_num = []
			infos = []


			for i, t_index in enumerate(cur_batch):
				pos_ins = self.test_index[t_index]
				index = pos_ins['index']
				h_idx = pos_ins['h_idx']
				t_idx = pos_ins['t_idx']
				r = pos_ins['r_idx']

				ins = self.test_file[index]
				Ls = ins['Ls']
				max_sents = max(max_sents, len(Ls) - 1)
				infos.append((ins['title'], h_idx, t_idx, self.id2rel[r]))


				context_idxs[i].copy_(torch.from_numpy(self.data_test_word[index, :]))
				context_char_idxs[i].copy_(torch.from_numpy(self.data_test_char[index, :]))
				context_ner[i].copy_(torch.from_numpy(self.data_test_ner[index, :]))
				relation_label[i] = r

				hlist = ins['vertexSet'][h_idx]
				tlist = ins['vertexSet'][t_idx]

				for h in hlist:
					context_pos[i, h['pos'][0]:h['pos'][1]] = 1

				for t in tlist:
					context_pos[i, t['pos'][0]:t['pos'][1]] = 2


				evidence = []
				for label in ins['labels']:
					if (label['h'], label['t'], label['r']) == (h_idx, t_idx, r):
						evidence = [int(e) for e in label['evidence']]

				evidences.append(evidence)

				for j in range(len(Ls) - 1):
					sent_h_mapping[i, j, Ls[j]] = 1
					sent_t_mapping[i, j, Ls[j + 1] - 1] = 1
					sent_mask[i, j] = 1

				sents_num.append(len(Ls)-1)

			input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
			max_c_len = int(input_lengths.max())

			yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
				   'relation_label': relation_label[:cur_bsz].contiguous(),
				   'input_lengths' : input_lengths,
				   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
				   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'sent_h_mapping': sent_h_mapping[:cur_bsz, :max_sents, :max_c_len],
				   'sent_t_mapping': sent_t_mapping[:cur_bsz, :max_sents, :max_c_len],
				   'sent_mask': sent_mask[:cur_bsz, :max_sents],
				   'evidences': evidences,
				   'sents_num': sents_num,
				   'infos': infos
				   }


	def train(self, model_pattern, model_name):
		ori_model = model_pattern(config = self)
		if self.pretrain_model != None:
			ori_model.load_state_dict(torch.load(self.pretrain_model))
		ori_model.cuda()
		model = nn.DataParallel(ori_model)

		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

		BCE = nn.BCEWithLogitsLoss(reduction='none')

		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)

		best_auc = 0.0
		best_f1 = 0.0
		best_epoch = 0

		model.train()

		global_step = 0
		total_loss = 0
		start_time = time.time()

		def logging(s, print_=True, log_=True):
			if print_:
				print(s)
			if log_:
				with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
					f_log.write(s + '\n')

		for epoch in range(self.max_epoch):

			self.acc_NA.clear()
			self.acc_not_NA.clear()
			self.acc_total.clear()

			for data in self.get_N2_train_batch():

				context_idxs = data['context_idxs']
				context_pos = data['context_pos']
				relation_label = data['relation_label']
				input_lengths =  data['input_lengths']
				context_ner = data['context_ner']
				context_char_idxs = data['context_char_idxs']
				sent_h_mapping = data['sent_h_mapping']
				sent_t_mapping = data['sent_t_mapping']
				sent_mask = data['sent_mask']
				evidence_label = data['evidence_label']

				predict_sent = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths, sent_h_mapping, sent_t_mapping, relation_label)
				loss = torch.sum(BCE(predict_sent, evidence_label) * sent_mask) / torch.sum(sent_mask)


				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				global_step += 1
				total_loss += loss.item()

				if global_step % self.period == 0 :
					cur_loss = total_loss / self.period
					elapsed = time.time() - start_time
					logging('| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f} '.format(epoch, global_step, elapsed * 1000 / self.period, cur_loss))
					total_loss = 0
					start_time = time.time()



			if (epoch + 1) % self.test_epoch == 0:
				logging('-' * 89)
				eval_start_time = time.time()
				model.eval()
				f1 = self.test(model, model_name)
				model.train()
				logging('| epoch {:3d} | time: {:5.2f}s | F1 {:.4f}'.format(epoch, time.time() - eval_start_time, f1))
				logging('-' * 89)


				if f1 > best_f1:
					best_f1 = f1
					best_epoch = epoch
					path = os.path.join(self.checkpoint_dir, model_name)
					torch.save(ori_model.state_dict(), path)

		print("Finish training")
		print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
		print("Storing best result...")
		print("Finish storing")

	def test(self, model, model_name, output=False, input_theta=-1):
		test_evidence_result = []

		def logging(s, print_=True, log_=True):
			if print_:
				print(s)
			if log_:
				with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
					f_log.write(s + '\n')

		for data in self.get_real_test_batch():
			with torch.no_grad():
				context_idxs = data['context_idxs']
				context_pos = data['context_pos']
				relation_label = data['relation_label']
				input_lengths = data['input_lengths']
				context_ner = data['context_ner']
				context_char_idxs = data['context_char_idxs']
				sent_h_mapping = data['sent_h_mapping']
				sent_t_mapping = data['sent_t_mapping']
				evidences = data['evidences']
				sents_num = data['sents_num']
				infos = data['infos']


				predict_sent = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths, sent_h_mapping, sent_t_mapping, relation_label)

				predict_sent = torch.sigmoid(predict_sent)


			predict_sent = predict_sent.data.cpu().numpy()

			for i in range(len(evidences)):
				evi = evidences[i]
				for j in range(sents_num[i]):
					test_evidence_result.append( (j in evi, float(predict_sent[i, j]), infos[i], j) )



		test_evidence_result.sort(key = lambda x: x[1], reverse=True)

		total_evidence_recall = self.total_evidence_recall
		if total_evidence_recall==0:   # for test
			total_evidence_recall = 1

		pr_x = []
		pr_y = []
		correct = 0
		w = 0

		for i, item in enumerate(test_evidence_result):
			correct += item[0]
			pr_y.append(float(correct) / (i + 1))
			pr_x.append(float(correct) / total_evidence_recall)
			if item[1] > input_theta:
				w = i


		pr_x = np.asarray(pr_x, dtype='float32')
		pr_y = np.asarray(pr_y, dtype='float32')
		f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
		f1_pos = f1_arr.argmax()
		evidence_f1 = f1_arr.max()
		auc = sklearn.metrics.auc(x = pr_x, y = pr_y)

		if input_theta==-1:
			w = f1_pos
			input_theta = test_evidence_result[w][1]

		logging('ma_f1{:3.4f} | input_theta {:3.4f} test_evidence_result F1 {:3.4f} | AUC {:3.4f}'.format(evidence_f1, input_theta, f1_arr[w], auc))

		if output:
			info2evi = {}

			for x in self.test_index:
				info2evi[(x['title'], x['h_idx'], x['t_idx'], x['r'])] = []


			for i in range(w+1):
				info = test_evidence_result[i][-2]
				sent_id = test_evidence_result[i][-1]
				info2evi[info].append(sent_id)


			output = []
			for u, v in info2evi.items():
				title = u[0]
				h_idx = u[1]
				t_idx = u[2]
				r = u[3]
				evidence = v
				output.append({'title':title, 'h_idx': h_idx, 't_idx': t_idx, 'r': r, 'evidence': evidence})

			json.dump(output, open(self.output_file, "w"))

		return evidence_f1



	def testall(self, model_pattern, model_name, input_theta=-1):
		model = model_pattern(config = self)

		model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
		model.cuda()
		model.eval()
		self.test(model, model_name, True, input_theta)

