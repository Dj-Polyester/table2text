DATA_DIR=raw_data/lyrics_stanzas
FILE_ID=1y46IMOa_oB9K-uD8gVmsGLcz6RQCPD9_
OUTPUT_TAR=lyrics_stanzas_split.tar.gz
COOKIE_JAR=/tmp/cookies.txt

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"
pushd "${DATA_DIR}"

TMP_HTML=$(mktemp)
wget --quiet --save-cookies "${COOKIE_JAR}" --keep-session-cookies --no-check-certificate \
  "https://docs.google.com/uc?export=download&id=${FILE_ID}" \
  -O "${TMP_HTML}"

CONFIRM_TOKEN=$(sed -rn 's/.*confirm=([0-9A-Za-z_-]+).*/\1/p; q' "${TMP_HTML}")
if [ -n "${CONFIRM_TOKEN}" ]; then
  wget --load-cookies "${COOKIE_JAR}" \
    "https://docs.google.com/uc?export=download&confirm=${CONFIRM_TOKEN}&id=${FILE_ID}" \
    -O "${OUTPUT_TAR}"
else
  FORM_ACTION=$(sed -rn 's/.*<form[^>]*action="([^"]+)".*/\1/p; q' "${TMP_HTML}")
  FORM_CONFIRM=$(sed -rn 's/.*name="confirm" value="([^"]+)".*/\1/p; q' "${TMP_HTML}")
  FORM_UUID=$(sed -rn 's/.*name="uuid" value="([^"]+)".*/\1/p; q' "${TMP_HTML}")
  if [ -z "${FORM_ACTION}" ] || [ -z "${FORM_CONFIRM}" ]; then
    echo "Could not extract Google Drive confirmation form values" >&2
    rm -f "${COOKIE_JAR}" "${TMP_HTML}"
    popd
    exit 1
  fi
  if [[ "${FORM_ACTION}" != http* ]]; then
    FORM_ACTION="https://drive.usercontent.google.com${FORM_ACTION}"
  fi
  DOWNLOAD_URL="${FORM_ACTION}?id=${FILE_ID}&export=download&confirm=${FORM_CONFIRM}"
  if [ -n "${FORM_UUID}" ]; then
    DOWNLOAD_URL="${DOWNLOAD_URL}&uuid=${FORM_UUID}"
  fi
  wget --load-cookies "${COOKIE_JAR}" "${DOWNLOAD_URL}" -O "${OUTPUT_TAR}"
fi

rm -f "${COOKIE_JAR}" "${TMP_HTML}"

if ! tar -tzf "${OUTPUT_TAR}" >/dev/null 2>&1; then
  echo "Downloaded file is not a valid tar.gz archive" >&2
  exit 1
fi

tar xvfz "${OUTPUT_TAR}"
rm -f *.tar.gz
sha256sum *.txt
popd
