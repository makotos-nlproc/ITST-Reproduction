# mustc_root=PATH_TO_MUSTC_DATA
mustc_root=~/ITST-Reproduction/data
lang=de

# mkdir en-de-eval
eval_data~/ITST-Reproduction/data/en-de-eval

# unzip
# tar -xzvf ${mustc_root}/MUSTC_v1.0_en-${lang}.tar.gz

export PYTHONPATH="~/ITST-reproduction/examples:$PYTHONPATH"

# prepare ASR data
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${mustc_root} --task asr \
  --vocab-type unigram --vocab-size 10000 \
  --cmvn-type global

# prepare ST data
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${mustc_root} --task st \
  --vocab-type unigram --vocab-size 10000 \
  --cmvn-type global

# generate the wav list and reference file for SimulEval
for split in dev tst-COMMON tst-HE
do
    python examples/speech_to_text/seg_mustc_data.py \
    --data-root ${mustc_root} --lang ${lang} \
    --split ${split} --task st \
    --output ${eval_data}/${split}
done