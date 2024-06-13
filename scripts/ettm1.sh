if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=FTMixer
root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2021

seq_len=336
for pred_len in   336 720
do
    CUDA_VISIBLE_DEVICES=4 \
  python -u run_longExp.py \
    --random_seed 2021 \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm1.csv \
    --model_id ETTm1'_'$seq_len'_'$pred_len \
    --model FTMixer \
    --data ETTm1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 1 \
    --d_model 32 \
    --d_ff 64 \
    --dropout 0.3\
    --fc_dropout 0.2 \
    --kernel_list 3 7 9 11 \
    --period 48 84    \
    --patch_len 1 1    \
    --stride 1 1    \
    --des Exp \
    --train_epochs 20 \
    --patience 5 \
    --itr 1 --batch_size 512 --learning_rate 0.005 |tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done