CUDA_VISIBLE_DEVICES=4,5,6,7 python run_clm.py \
--model_name_or_path distilgpt2 \
--run_name wechsel-gpt2-ko \
--dataset_name oscar \
--dataset_config_name unshuffled_deduplicated_ko \
--output_dir ./wechsel-gpt2-ko \
--save_steps 1000 \
--save_total_limit 10 \
--report_to all \
--logging_first_step \
--logging_steps 100 \
--do_train \
--do_eval \
--validation_split_percentage 5 \
--fp16 true \
--num_train_epochs 2 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 16
