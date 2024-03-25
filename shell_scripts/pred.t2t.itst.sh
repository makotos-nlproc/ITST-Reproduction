export PYTHONPATH="~/ITST-reproduction/examples:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

data=~/ITST-Reproduction/data

modelfile=~/ITST-Reproduction/models/st_model/EXP-003
last_file=checkpoint_last.pt

eval_root=~/ITST-Reproduction/data/en-de-eval/tst-COMMON
ref_dir=${eval_root}/tst-COMMON.wav_list
detok_ref_dir=${eval_root}/tst-COMMON.de

output_dir=~/ITST-Reproduction/models/st_model_eval/gen

# mosesdecoder=PATH_TO_MOSESD # https://github.com/moses-smt/mosesdecoder
src_lang=en
tgt_lang=de

threshold=0.8 # test threshold in ITST, such as 0.8

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 \
    --output ${output_dir}/average-model.pt --last_file ${last_file}
file=${output_dir}/average-model.pt 

# generate translation
python fairseq_cli/sim_generate.py ${data} --path ${file} \
    --batch-size 1 --beam 1 --left-pad-source --fp16 --remove-bpe \
    --itst-decoding --itst-test-threshold ${threshold} > pred.out 2>&1

# latency
echo -e "\nLatency"
tail -n 4 pred.out

# BLEU
# echo -e "\nBLEU"
# grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
# multi-bleu.perl -lc ${ref_dir} < pred.translation

# SacreBLEU
# echo -e "\nSacreBLEU"
# perl ${mosesdecoder}/scripts/tokenizer/detokenizer.perl -l ${tgt_lang} < pred.translation > pred.translation.detok
# cat pred.translation.detok | sacrebleu ${detok_ref_dir} --w 2
