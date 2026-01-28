devices=0
llm=bigcode/starcoder2-7b
model=starcoder2-7b
maxlength=630
lr=1e-4
project_path=/root/autodl-tmp/VulCure
batchsize=2

data_type=random
train_dataset=vulcure_vd_${data_type}_trainplus_800



modelwrapper=mle
dr=0
#

for epoch in 3
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset $train_dataset \
    --model-type causallm --modelwrapper $modelwrapper  --lr $lr --batch-size $batchsize --opt adamw --warmup-ratio 0.1  \
    --max-seq-len $maxlength \
     --nowand  \
     --lora-r 8 --lora-alpha 16  --model $llm  \
    --load-model-path  $project_path/my_llm/$llm --lora-dropout $dr  --checkpoint --seed 4321 \
     --n-epochs $epoch  --checkpoint-dic-name epoch$epoch/$model-1
done

for dataset in vulcure_vd_${data_type}_test_800 vulcure_vd_${data_type}_correct_800 vulcure_vd_${data_type}_train_800
do
for epoch in 3
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset $train_dataset \
    --ood-ori-dataset $dataset --model-type causallm --model $llm --modelwrapper $modelwrapper \
    --batch-size $batchsize   --max-seq-len $maxlength  --seed 4321 \
    --lora-dropout $dr   \
    --nowand --load-model-path $project_path/my_llm/$llm \
    --bayes-eval-index 1 --n-epochs $epoch \
    --load-lora-path  $project_path/bayesian_peft/checkpoints/causallm_$modelwrapper/$llm/$train_dataset/epoch$epoch/$model-1
done
done


modelwrapper=blob
dr=0
#

for epoch in 3
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset $train_dataset \
    --model-type causallm --modelwrapper $modelwrapper  --lr $lr --batch-size $batchsize --opt adamw --warmup-ratio 0.1  \
    --max-seq-len $maxlength \
     --nowand  \
     --lora-r 8 --lora-alpha 16  --model $llm  \
    --load-model-path  $project_path/my_llm/$llm --lora-dropout $dr  --checkpoint --seed 4321 \
     --n-epochs $epoch  --checkpoint-dic-name epoch$epoch/$model-1
done

for dataset in vulcure_vd_${data_type}_test_800 vulcure_vd_${data_type}_correct_800 vulcure_vd_${data_type}_train_800
do
for epoch in 3
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset $train_dataset \
    --ood-ori-dataset $dataset --model-type causallm --model $llm --modelwrapper $modelwrapper \
    --batch-size $batchsize   --max-seq-len $maxlength  --seed 4321 \
    --lora-dropout $dr   \
    --nowand --load-model-path $project_path/my_llm/$llm \
    --bayes-eval-index 1 --n-epochs $epoch \
    --load-lora-path  $project_path/bayesian_peft/checkpoints/causallm_$modelwrapper/$llm/$train_dataset/epoch$epoch/$model-1
done
done


modelwrapper=mcdropout
dr=0.1
#

for epoch in 3
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset $train_dataset \
    --model-type causallm --modelwrapper $modelwrapper  --lr $lr --batch-size $batchsize --opt adamw --warmup-ratio 0.1  \
    --max-seq-len $maxlength \
     --nowand  \
     --lora-r 8 --lora-alpha 16  --model $llm  \
    --load-model-path  $project_path/my_llm/$llm --lora-dropout $dr  --checkpoint --seed 4321 \
     --n-epochs $epoch  --checkpoint-dic-name epoch$epoch/$model-1
done

for dataset in vulcure_vd_${data_type}_test_800 vulcure_vd_${data_type}_correct_800 vulcure_vd_${data_type}_train_800
do
for epoch in 3
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset $train_dataset \
    --ood-ori-dataset $dataset --model-type causallm --model $llm --modelwrapper $modelwrapper \
    --batch-size $batchsize   --max-seq-len $maxlength  --seed 4321 \
    --lora-dropout $dr   \
    --nowand --load-model-path $project_path/my_llm/$llm \
    --bayes-eval-index 1 --n-epochs $epoch \
    --load-lora-path  $project_path/bayesian_peft/checkpoints/causallm_$modelwrapper/$llm/$train_dataset/epoch$epoch/$model-1
done
done



dr=0
modelwrapper=deepensemble

for epoch in 3
do
    CUDA_VISIBLE_DEVICES=$devices python run/main.py --dataset-type mcdataset --dataset $train_dataset \
    --model-type causallm --modelwrapper $modelwrapper  --lr $lr --batch-size $batchsize --opt adamw --warmup-ratio 0.1  \
    --max-seq-len $maxlength \
     --nowand  \
     --lora-r 8 --lora-alpha 16  --model $llm  --ensemble-n 10 \
    --load-model-path $project_path/my_llm/$llm --lora-dropout $dr  --checkpoint --seed 1234 \
     --n-epochs $epoch  --checkpoint-dic-name epoch$epoch/$model-1
done

for dataset in  vulcure_vd_${data_type}_test_800 vulcure_vd_${data_type}_correct_800 vulcure_vd_${data_type}_train_800
do
for epoch in 3
do
    CUDA_VISIBLE_DEVICES=$devices python run/main.py --dataset-type mcdataset --dataset $train_dataset \
    --ood-ori-dataset $dataset --model-type causallm --model $llm --modelwrapper $modelwrapper \
    --batch-size $batchsize  --max-seq-len $maxlength  --seed 1234 \
    --lora-dropout $dr  \
    --nowand --load-model-path /home/gpu_user/my_llm/$llm \
     --n-epochs $epoch  --ensemble-n 10 \
    --load-lora-path  $project_path/bayesian_peft/checkpoints/causallm_$modelwrapper/$llm/$train_dataset/epoch$epoch/$model-1
done
done





