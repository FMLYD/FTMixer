if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=FTMixer
root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
random_seed=2024
seq_len=336

for pred_len in     336 720
do
  python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 3 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2'_'$seq_len'_'$pred_len \
  --model FTMixer \
  --data ETTh2 \
  --features M \
  --gpu 4\
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 4 \
  --d_model 32\
  --d_ff 64\
  --dropout 0.5 \
  --fc_dropout 0 \
  --kernel_list 3 7 9 11\
  --period 24  \
  --patch_len 1  \
  --m_layers 0\
  --m_model 336\
  --stride 1  \
  --des Exp \
  --train_epochs 20\
  --patience 3 \
  --itr 1 --batch_size 256 --learning_rate 0.001 |tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done