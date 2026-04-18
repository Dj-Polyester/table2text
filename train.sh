DATASET=wiki_bio
# DATASET=arxiv_cs_abstracts
TRAIN_DIR=train
EXAMPLES_DIR=data/char_masks/${DATASET}
python train_ilm.py \
	experiment_${DATASET} \
	${TRAIN_DIR} \
	${EXAMPLES_DIR} \
	--seed 0 \
	--tensorboard \
	--train_examples_tag train \
	--eval_examples_tag val \
	--data_loader_num_workers 16