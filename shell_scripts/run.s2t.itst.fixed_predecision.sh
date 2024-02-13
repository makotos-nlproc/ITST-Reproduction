export PYTHONPATH="~/ITST-reproduction/examples:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# TODO: check arg for exp_name
EXP_NAME=$1

mustc_root=~/ITST-Reproduction/data
lang=de
asr_modelfile=~/ITST-Reproduction/asr_model/${EXP_NAME}
# mkdir st_model
st_modelfile=~/ITST-Reproduction/st_model/${EXP_NAME}
mkdir -p ${st_modelfile}
pretrain_model=checkpoint_best.pt

python train.py ${mustc_root}/en-${lang} \
    --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
    --user-dir examples/simultaneous_translation \
    --save-dir ${st_modelfile} --num-workers 8  \
    --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
    --criterion label_smoothed_cross_entropy_with_itst_s2t_fixed_predecision \
    --warmup-updates 4000 --max-update 300000 --max-tokens 20000 --seed 2 \
    --label-smoothing 0.1 \
    --load-pretrained-encoder-from ${asr_modelfile}/${pretrain_model} \
    --task speech_to_text  \
    --arch convtransformer_simul_trans_itst_espnet  \
    --unidirectional-encoder \
    --simul-type ITST_fixed_pre_decision  \
    --fixed-pre-decision-ratio 7 \
    --update-freq 4 \
    --save-interval-updates 1000 \
    --keep-interval-updates 200 \
    --find-unused-parameters \
    --fp16 \
    --log-interval 10 \
    --wandb-project itst-st-train-${EXP_NAME}