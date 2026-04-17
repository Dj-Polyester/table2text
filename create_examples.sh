DATASET=michaelauli/wiki_bio

DATA_DIR_PARENT=data/raw_data
MASKDIR="${DATASET}"

if [[ " arxiv_cs_abstracts lyrics_stanzas roc_stories " == *" ${DATASET} "* ]]; then
	pushd data
	./get_${DATASET}.sh
	popd
	DATA_DIR="${DATA_DIR_PARENT}/${DATASET}"
elif [ -d "${DATA_DIR_PARENT}/${DATASET}" ]; then
	DATA_DIR="${DATA_DIR_PARENT}/${DATASET}"
else
	DATA_DIR="${DATASET}"
	MASKDIR="$(basename ${DATASET})"
	DATASET=hf
fi

for SPLIT in train val
do
	python create_ilm_examples.py \
		${SPLIT} \
		data/char_masks/${MASKDIR} \
		--seed 0 \
		--data_name ${DATASET} \
		--data_dir ${DATA_DIR} \
		--max_num_documents 100 \
		--data_split ${SPLIT}
done

