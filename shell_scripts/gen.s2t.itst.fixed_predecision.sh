export PYTHONPATH="~/ITST-reproduction/examples:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

mustc_root=~/ITST-Reproduction/data

modelfile=~/ITST-Reproduction/models/st_model/EXP-003
last_file=checkpoint_last.pt

eval_root=~/ITST-Reproduction/data/en-de-eval/tst-COMMON
ref_dir=${eval_root}/tst-COMMON.wav_list
detok_ref_dir=${eval_root}/tst-COMMON.de

output_dir=~/ITST-Reproduction/models/st_model_eval/gen

itst_threshold=0.8 # test threshold in ITST, such as 0.8

# average last 5 checkpoints
# python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 \
#     --output ${output_dir}/average-model.pt --last_file ${last_file}

average_model_path=${output_dir}/average-model.pt 

# generate translation
python fairseq_cli/sim_generate_s2t.py ${mustc_root}/en-de \
  --task speech_to_text \
  --config-yaml config_st.yaml \
  --user-dir examples/simultaneous_translation \
  --gen-subset tst-COMMON_st \
  --path ${average_model_path} \
  --max-tokens 20000 \
  --beam 1 --scoring wer \
  --simul-type ITST_fixed_pre_decision \
  --fixed-pre-decision-ratio 7 \
  --itst-decoding --itst-test-threshold ${itst_threshold}