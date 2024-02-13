export PYTHONPATH="~/ITST-reproduction/examples:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

EXP_NAME=$1

mustc_root=~/ITST-Reproduction/data
lang=de
modelfile=~/ITST-Reproduction/asr_model/${EXP_NAME}
last_file=${modelfile}/checkpoint_last.pt

# average last 5 checkpoints
n=5
python scripts/average_checkpoints.py \
  --inputs ${modelfile} \
  --num-update-checkpoints ${n} \
  --output ${modelfile}/average-model.pt \
  --last_file ${last_file}

file=${modelfile}/average-model.pt 

python fairseq_cli/generate.py ${mustc_root}/en-${lang} \
  --config-yaml config_asr.yaml \
  --gen-subset tst-COMMON_asr \
  --task speech_to_text \
  --path ${file} \
  --max-tokens 50000 \
  --max-source-positions 6000 \
  --beam 1 \
  --scoring wer \
  --wer-tokenizer 13a \
  --wer-lowercase --wer-remove-punct