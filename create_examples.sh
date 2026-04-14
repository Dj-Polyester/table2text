DATASET=arxiv_cs_abstracts

if [ "${DATASET}" != "wiki_bio" ]; then
	pushd data
	./get_${DATASET}.sh
	popd
fi

for SPLIT in train val
do
	python create_ilm_examples.py \
		${SPLIT} \
		data/char_masks/${DATASET} \
		--seed 0 \
		--data_name ${DATASET} \
		--max_num_documents 100 \
		--data_split ${SPLIT}
done

