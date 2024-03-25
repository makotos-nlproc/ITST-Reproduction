export PYTHONPATH="~/ITST-reproduction/examples:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

EXP_NAME=001

MUSTC_ROOT=~/ITST-Reproduction/data
lang=de

ASR_SAVE_DIR=~/ITST-Reproduction/models/asr_model/EXP-003
ST_SAVE_DIR=~/ITST-Reproduction/models/simulst_mustc_example/EXP-${EXP_NAME}

python train.py ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${ST_SAVE_DIR} --num-workers 8  \
  --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
  --criterion label_smoothed_cross_entropy \
  --warmup-updates 4000 --max-update 100000 --max-tokens 40000 --seed 2 \
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}/checkpoint_best.pt \
  --task speech_to_text  \
  --arch convtransformer_simul_trans_espnet  \
  --simul-type waitk_fixed_pre_decision  \
  --waitk-lagging 3 \
  --fixed-pre-decision-ratio 7 \
  --update-freq 8
  ----wandb-project simulst-mustc-example-${EXP_NAME}
