if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=FTMixer

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2021

for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 1 \
      --n_heads 32 \
      --d_model 64 \
      --d_ff 256\
      --dropout 0.45 \
      --fc_dropout 0.15 \
      --kernel_list 3 3 5\
      --period 8  \
      --patch_len 1  \
      --stride 1  \
      --des 'Exp'\
      --train_epochs 10\
      --pct_start 0.2\
      --m_layers 0 --m_model 32\
      --patience 3\
      --itr 1 --batch_size 1 --learning_rate 0.004 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
