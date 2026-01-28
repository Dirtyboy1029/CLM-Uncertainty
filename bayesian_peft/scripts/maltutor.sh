devices=0
llm=bigcode/starcoder2-7b
model=starcoder2-7b
maxlength=630
lr=1e-4
data_type=dataset
project_path=/root/autodl-tmp/VulCure
batchsize=2
seed=4321

modelwrapper=mle
dr=0.0
llm_safe=${llm//\//_}


for epoch in 1
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset vulcure_vd_${data_type}_easy_${llm_safe} \
    --model-type causallm --modelwrapper $modelwrapper  --lr $lr --batch-size $batchsize --opt adamw --warmup-ratio 0.1  \
    --max-seq-len $maxlength \
     --nowand  \
     --lora-r 8 --lora-alpha 16  --model $llm  \
    --load-model-path $project_path/my_llm/$llm --lora-dropout $dr  --checkpoint --seed $seed \
     --n-epochs $epoch  --checkpoint-dic-name epoch$epoch/$model-maltutor-easy
done

for epoch in 1
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset vulcure_vd_${data_type}_diff_${llm_safe} \
    --model-type causallm --modelwrapper $modelwrapper  --lr $lr --batch-size $batchsize --opt adamw --warmup-ratio 0.1  \
    --max-seq-len $maxlength \
     --nowand  --load-checkpoint \
     --load-model-path ${project_path}/bayesian_peft/checkpoints/causallm_mle/${llm}/vulcure_vd_${data_type}_easy_${llm_safe}/epoch1/${model}-maltutor-easy \
     --lora-r 8 --lora-alpha 16  --model $llm  \
     --lora-dropout $dr  --checkpoint --seed $seed \
     --n-epochs $epoch  --checkpoint-dic-name epoch$epoch/$model-maltutor-easy-diff
done


for epoch in 1
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset vulcure_vd_${data_type}_correct_800 \
    --model-type causallm --modelwrapper $modelwrapper  --lr $lr --batch-size $batchsize --opt adamw --warmup-ratio 0.1  \
    --max-seq-len $maxlength \
     --nowand  --load-checkpoint \
     --load-model-path ${project_path}/bayesian_peft/checkpoints/causallm_mle/${llm}/vulcure_vd_${data_type}_diff_${llm_safe}/epoch1/${model}-maltutor-easy-diff \
     --lora-r 8 --lora-alpha 16  --model $llm  \
     --lora-dropout $dr  --checkpoint --seed $seed \
     --n-epochs $epoch  --checkpoint-dic-name epoch$epoch/$model-maltutor-easy-diff-correct
done


traindata_name=vulcure_vd_${data_type}_easy_${llm_safe}
suffix=maltutor-easy
for dataset in  vulcure_vd_${data_type}_test_800
do
for epoch in 1
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset  $traindata_name \
    --ood-ori-dataset $dataset --model-type causallm --model $llm --modelwrapper $modelwrapper \
    --batch-size $batchsize   --max-seq-len $maxlength  --seed $seed \
    --lora-dropout $dr  \
    --nowand --load-model-path $project_path/my_llm/$llm \
    --bayes-eval-index $suffix --n-epochs $epoch \
    --load-lora-path  $project_path/bayesian_peft/checkpoints/causallm_$modelwrapper/$llm/${traindata_name}/epoch$epoch/${model}-${suffix}
done
done

traindata_name=vulcure_vd_${data_type}_diff_${llm_safe}
suffix=maltutor-easy-diff
for dataset in  vulcure_vd_${data_type}_test_800
do
for epoch in 1
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset $traindata_name \
    --ood-ori-dataset $dataset --model-type causallm --model $llm --modelwrapper $modelwrapper \
    --batch-size $batchsize   --max-seq-len $maxlength  --seed $seed \
    --lora-dropout $dr  \
    --nowand --load-model-path $project_path/my_llm/$llm \
    --bayes-eval-index $suffix --n-epochs $epoch \
    --load-lora-path  $project_path/bayesian_peft/checkpoints/causallm_$modelwrapper/$llm/${traindata_name}/epoch$epoch/${model}-${suffix}
done
done


traindata_name=vulcure_vd_${data_type}_correct_800
suffix=maltutor-easy-diff-correct
for dataset in  vulcure_vd_${data_type}_test_800
do
for epoch in 1
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset $traindata_name \
    --ood-ori-dataset $dataset --model-type causallm --model $llm --modelwrapper $modelwrapper \
    --batch-size $batchsize   --max-seq-len $maxlength  --seed $seed \
    --lora-dropout $dr  \
    --nowand --load-model-path $project_path/my_llm/$llm \
    --bayes-eval-index $suffix --n-epochs $epoch \
    --load-lora-path  $project_path/bayesian_peft/checkpoints/causallm_$modelwrapper/$llm/${traindata_name}/epoch$epoch/${model}-${suffix}
done
done




##########################################
for epoch in 1
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset vulcure_vd_${data_type}_train_800 \
    --model-type causallm --modelwrapper $modelwrapper  --lr $lr --batch-size $batchsize --opt adamw --warmup-ratio 0.1  \
    --max-seq-len $maxlength \
     --nowand  --load-checkpoint \
     --load-model-path ${project_path}/bayesian_peft/checkpoints/causallm_mle/${llm}/vulcure_vd_${data_type}_easy_${llm_safe}/epoch1/${model}-maltutor-easy \
     --lora-r 8 --lora-alpha 16  --model $llm  \
     --lora-dropout $dr  --checkpoint --seed $seed \
     --n-epochs $epoch  --checkpoint-dic-name epoch$epoch/$model-maltutor-easy-train
done
#
#
for epoch in 1
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset vulcure_vd_${data_type}_trainplus_800 \
    --model-type causallm --modelwrapper $modelwrapper  --lr $lr --batch-size $batchsize --opt adamw --warmup-ratio 0.1  \
    --max-seq-len $maxlength \
     --nowand  --load-checkpoint \
     --load-model-path ${project_path}/bayesian_peft/checkpoints/causallm_mle/${llm}/vulcure_vd_${data_type}_train_800/epoch1/${model}-maltutor-easy-train \
     --lora-r 8 --lora-alpha 16  --model $llm  \
     --lora-dropout $dr  --checkpoint --seed $seed \
     --n-epochs $epoch  --checkpoint-dic-name epoch$epoch/$model-maltutor-easy-train-trainplus
done


traindata_name=vulcure_vd_${data_type}_train_800
suffix=maltutor-easy-train
for dataset in  vulcure_vd_${data_type}_test_800
do
for epoch in 1
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset $traindata_name \
    --ood-ori-dataset $dataset --model-type causallm --model $llm --modelwrapper $modelwrapper \
    --batch-size $batchsize   --max-seq-len $maxlength  --seed $seed \
    --lora-dropout $dr  \
    --nowand --load-model-path $project_path/my_llm/$llm \
    --bayes-eval-index $suffix --n-epochs $epoch \
    --load-lora-path  $project_path/bayesian_peft/checkpoints/causallm_$modelwrapper/$llm/${traindata_name}/epoch$epoch/${model}-${suffix}
done
done


traindata_name=vulcure_vd_${data_type}_trainplus_800
suffix=maltutor-easy-train-trainplus
for dataset in  vulcure_vd_${data_type}_test_800
do
for epoch in 1
do
    CUDA_VISIBLE_DEVICES=$devices python3 run/main.py --dataset-type mcdataset --dataset $traindata_name \
    --ood-ori-dataset $dataset --model-type causallm --model $llm --modelwrapper $modelwrapper \
    --batch-size $batchsize   --max-seq-len $maxlength  --seed $seed \
    --lora-dropout $dr  \
    --nowand --load-model-path $project_path/my_llm/$llm \
    --bayes-eval-index $suffix --n-epochs $epoch \
    --load-lora-path  $project_path/bayesian_peft/checkpoints/causallm_$modelwrapper/$llm/${traindata_name}/epoch$epoch/${model}-${suffix}
done
done





