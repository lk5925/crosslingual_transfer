CUDA_VISIBLE_DEVICES=0 python run_clm.py \
--model_name_or_path gpt2 \
--run_name wechsel-gpt2-ko \
# --dataset_name oscar \
# --dataset_config_name unshuffled_deduplicated_ko \
--train_file ./kcbert-data/kcbert-data.txt \
--output_dir ./wechsel-gpt2-ko \
--evaluation_strategy steps \
--overwrite_output_dir \
--eval_steps 1000 \
--save_steps 1000 \
--save_total_limit 10 \
--report_to all \
--logging_first_step \
--logging_steps 1000 \
--do_train \
--do_eval \
--validation_split_percentage 5 \
--load_best_model_at_end true \
--metric_for_best_model ppl \
--fp16 true \
--num_train_epochs 200 \
--gradient_accumulation_steps 1 \
--per_device_train_batch_size 16
