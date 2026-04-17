DATASET=michaelauli/wiki_bio

CHAR_MASKS_DIR_PARENT=data/char_masks
DATA_DIR_PARENT=data/raw_data

if [[ " arxiv_cs_abstracts lyrics_stanzas roc_stories " == *" ${DATASET} "* ]] && [ ! -d "${DATA_DIR_PARENT}/${DATASET}" ]; then
	pushd data
	./get_${DATASET}.sh
	popd
	DATA_DIR="${DATA_DIR_PARENT}/${DATASET}"
	MASKDIR="${DATASET}"
elif [ -d "${DATA_DIR_PARENT}/${DATASET}" ]; then
	DATA_DIR="${DATA_DIR_PARENT}/${DATASET}"
	MASKDIR="${DATASET}"
else
	DATA_DIR="${DATASET}"
	MASKDIR="$(basename ${DATASET})"
	DATASET=hf
fi

for SPLIT in train val
do
	python create_ilm_examples.py \
		${SPLIT} \
		${CHAR_MASKS_DIR_PARENT}/${MASKDIR} \
		--seed 0 \
		--data_name ${DATASET} \
		--data_dir ${DATA_DIR} \
		--data_split ${SPLIT} \
		--num_workers 16
done

