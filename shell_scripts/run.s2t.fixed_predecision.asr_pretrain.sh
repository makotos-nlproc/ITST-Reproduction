export PYTHONPATH="~/ITST-reproduction/examples:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

EXP_NAME=$1

mustc_root=~/ITST-Reproduction/data
lang=de
# mkdir checkopoints
asr_modelfile=~/ITST-Reproduction/checkpoints/${EXP_NAME}
mkdir -p ${asr_modelfile}

python train.py ${mustc_root}/en-${lang} \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir ${asr_modelfile} --num-workers 4 --max-update 100000 --max-tokens 30000  \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch convtransformer_espnet --optimizer adam --lr 0.00005 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 3 \
  --save-interval-updates 1000 \
  --keep-interval-updates 100 \
  --find-unused-parameters \
  --fp16 \
  --log-interval 10 \
  --wandb-project itst-asr-pretrain-${EXP_NAME}