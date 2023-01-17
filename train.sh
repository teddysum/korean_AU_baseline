
python au_main.py \
  --train_data ../data/nikluge-au-2022-train.jsonl \
  --dev_data ../data/nikluge-au-2022-dev.jsonl \
  --base_model xlm-roberta-base \
  --do_train \
  --do_eval \
  --learning_rate 3e-6 \
  --eps 1e-8 \
  --num_train_epochs 10 \
  --model_path /root/data/saved_models/au_baseline/ \
  --batch_size 8 \
  --max_len 256