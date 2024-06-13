if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=FTMixer

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2024
for pred_len in 96 
do
for e_layers in  3
do
for m_layers in   0
do
for dropout in  0.3
do
for m_model in  16
do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
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
      --enc_in 862 \
      --e_layers $e_layers \
      --n_heads 1 \
      --d_model 128 \
      --fc_dropout 0.2\
      --d_ff 256 \
      --dropout $dropout\
      --period 8  12 24 \
      --patch_len 1 2 24 \
      --m_model $m_model\
      --stride 1 2 24 \
      --des 'Exp'\
      --train_epochs 100\
      --m_layers $m_layers\
      --patience 10\
      --pct_start 0.2\
      --use_multi_gpu --devices 0,1,2,3,4,5,6,7\
      --itr 1 --batch_size 512 --learning_rate 0.005 |tee logs/LongForecasting/$model_name'_next_'$model_id_name'_'$seq_len'_'$pred_len'_m_layers:'$m_layers'_e_layers:'$e_layers'_dropout:'$dropout'_m_model:'$m_model.log
done
done
done
done
done