#!/usr/bin/env python
import sys
import os
import os.path
import json

submission_answer_file = sys.argv[1]
truth_file = sys.argv[2]

truth = json.load(open(truth_file))
std = {}
tot_relations = 0
tot_evidences = 0
titleset = set([])
for x in truth:
    title = x['title']
    titleset.add(title)

    tot_relations += len(x['labels'])

    for label in x['labels']:
        r = label['r']
        h_idx = label['h']
        t_idx = label['t']
        std[(title, r, h_idx, t_idx)] = set(label['evidence'])
        tot_evidences += len(label['evidence'])



submission_answer = json.load(open(submission_answer_file))
correct_re = 0
correct_evidence = 0
pred_evi = 0
for x in submission_answer:
    title = x['title']
    h_idx = x['h_idx']
    t_idx = x['t_idx']
    r = x['r']

    if 'evidence' in x:
        evi = set(x['evidence'])
    else:
        evi = set([])
    pred_evi += len(evi)

    if (title, r, h_idx, t_idx) in std:
        correct_re += 1
        stdevi = std[(title, r, h_idx, t_idx)]
        correct_evidence += len(stdevi & evi)



re_p = 1.0 * correct_re / len(submission_answer)
re_r = 1.0 * correct_re / tot_relations
if re_p+re_r == 0:
    re_f1 = 0
else:
    re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

evi_p = 1.0 * correct_evidence / pred_evi if pred_evi>0 else 0
evi_r = 1.0 * correct_evidence / tot_evidences
if evi_p+evi_r == 0:
    evi_f1 = 0
else:
    evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

print ('RE_F1:', re_f1)
print ('Evi_F1:', evi_f1)

