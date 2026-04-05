DATASET=arxiv_cs_abstracts

pushd data
./get_${DATASET}.sh
popd

for SPLIT in train valid
do
	python create_ilm_examples.py \
		${SPLIT} \
		data/char_masks/${DATASET} \
		--seed 0 \
		--data_name ${DATASET} \
		--data_split ${SPLIT}
done

