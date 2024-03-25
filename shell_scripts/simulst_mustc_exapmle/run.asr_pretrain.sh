
# maybe same as run.s2t.fixed_predecision_asr_pretrain.sh

# fairseq-train ${MUSTC_ROOT}/en-de \
#   --config-yaml config_asr.yaml \
#   --train-subset train_asr \
#   --valid-subset dev_asr \
#   --save-dir ${ASR_SAVE_DIR} \
#   --num-workers 4 --max-tokens 40000 --max-update 100000 \
#   --task speech_to_text \
#   --criterion label_smoothed_cross_entropy \
#   --report-accuracy \
#   --arch convtransformer_espnet --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
#   --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
