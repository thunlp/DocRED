# DocRED
Dataset and code for baselines for [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127v3)

Multiple entities in a document generally exhibit complex inter-sentence relations, and cannot be well handled by existing relation extraction (RE) methods that typically focus on extracting intra-sentence relations for single entity pairs. In order to accelerate the research on document-level RE, we introduce DocRED, a new dataset constructed from Wikipedia and Wikidata with three features: 

+ DocRED annotates both named entities and relations, and is the largest human-annotated dataset for document-level RE from plain text.
+ DocRED requires reading multiple sentences in a document to extract entities and infer their relations by synthesizing all information of the document.
+ Along with the human-annotated data, we also offer large-scale distantly supervised data, which enables DocRED to be adopted for both supervised and weakly supervised scenarios.

## Codalab
If you are interested in our dataset, you are welcome to join in the Codalab competition at [DocRED](https://competitions.codalab.org/competitions/20717)


## Cite
If you use the dataset or the code, please cite this paper:
```
@inproceedings{yao2019DocRED,
  title={{DocRED}: A Large-Scale Document-Level Relation Extraction Dataset},
  author={Yao, Yuan and Ye, Deming and Li, Peng and Han, Xu and Lin, Yankai and Liu, Zhenghao and Liu, Zhiyuan and Huang, Lixin and Zhou, Jie and Sun, Maosong},
  booktitle={Proceedings of ACL 2019},
  year={2019}
}
```
