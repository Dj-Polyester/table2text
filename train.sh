DATASET=wiki_bio
TRAIN_DIR=train
EXAMPLES_DIR=data/char_masks/${DATASET}
python train_ilm.py \
	experiment_${DATASET} \
	${TRAIN_DIR} \
	${EXAMPLES_DIR} \
	--seed 0 \
	--train_examples_tag train \
	--eval_examples_tag val \
	--eval_max_num_examples 512 \
	--data_loader_num_workers 16