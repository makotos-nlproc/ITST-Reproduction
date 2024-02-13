export PYTHONPATH="~/ITST-reproduction/examples:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

mustc_root=~/ITST-Reproduction/data
lang=de
modelfile=~/ITST-Reproduction/multi_waitk_st_model/EXP-001
last_file=${modelfile}/checkpoint_last.pt

eval_root=~/ITST-Reproduction/data/en-de-eval/tst-COMMON
wav_list=${eval_root}/tst-COMMON.wav_list
reference=${eval_root}/tst-COMMON.de
output_dir=~/ITST-Reproduction/multi_waitk_st_model_eval

# test k in wait-k, such as 5
testk=5

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} \
  --num-update-checkpoints 5 \
  --output ${output_dir}/average-model.pt \
  --last_file ${last_file}

file=${output_dir}/average-model.pt 

simuleval --agent examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent.py \
    --source ${wav_list} \
    --target ${reference} \
    --data-bin ${mustc_root}/en-${lang} \
    --config config_st.yaml \
    --model-path ${file} \
    --test-waitk-lagging ${testk} \
    --output ${output_dir} \
    --scores --gpu \
    --port 1234