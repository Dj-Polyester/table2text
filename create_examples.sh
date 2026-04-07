DATASET=wiki_bio
LIMIT=""

while [ "$#" -gt 0 ]; do
	case "$1" in
		--limit)
			if [ -z "$2" ]; then
				echo "Error: --limit requires a value"
				exit 1
			fi
			LIMIT="$2"
			shift 2
			;;
		*)
			echo "Unknown argument: $1"
			echo "Usage: $0 [--limit N]"
			exit 1
			;;
	esac
done

if [ "${DATASET}" != "wiki_bio" ]; then
	pushd data
	./get_${DATASET}.sh
	popd
fi

for SPLIT in train val
do
	LIMIT_ARGS=()
	if [ -n "${LIMIT}" ]; then
		LIMIT_ARGS+=(--max_num_documents "${LIMIT}")
	fi

	python create_ilm_examples.py \
		${SPLIT} \
		data/char_masks/${DATASET} \
		--seed 0 \
		--data_name ${DATASET} \
		--data_split ${SPLIT} \
		"${LIMIT_ARGS[@]}"
done

