
python au_main.py \
  --test_data ../data/nikluge-au-2022-test.jsonl \
  --base_model xlm-roberta-base \
  --do_demo \
  --model_path /root/data/saved_models/au_baseline/saved_model_epoch_8.pt \
  --output_dir ./demo_output/ \
  --max_len 256