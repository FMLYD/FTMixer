if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=FTMixer

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
random_seed=2024

for seq_len in  336 
do
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
      --pred_len 96 \
      --enc_in 21 \
      --e_layers 3\
      --n_heads 32 \
      --d_model 128 \
      --d_ff 256 \
      --gpu 3\
      --dropout 0.3 \
      --period 67 336 \
      --patch_len 6 24 \
      --stride 6 24 \
      --des 'Exp' \
      --train_epochs 20 \
      --patience 3 \
      --pct_start 0.2 \
      --itr 1 --batch_size 128 --individual  --learning_rate 0.005 |tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 192 336
do
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
      --gpu 3\
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 32 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.5 \
      --period 67 96 \
      --patch_len 6 24 \
      --stride 6 24 \
      --des 'Exp' \
      --train_epochs 100 \
      --patience 15 \
      --pct_start 0.2 \
      --itr 1 --batch_size 256 --learning_rate 0.005 |tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


for pred_len in 720
do
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
      --enc_in 21 \
      --e_layers 3 \
      --gpu 3\
      --n_heads 32 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.6 \
      --persevere 336 \
      --period 67 96 \
      --patch_len 6 96 \
      --stride 6 24 \
      --des 'Exp' \
      --train_epochs 100 \
      --patience 15 \
      --pct_start 0.2 \
      --itr 1 --batch_size 256 --learning_rate 0.001 |tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
