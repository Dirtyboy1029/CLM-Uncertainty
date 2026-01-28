# CLM-Uncertainty

## Overview:
This code repository  is associated with our paper titled **Understanding and Exploiting Uncertainty in Code Language ModelsforVulnerability Detection (Experience Paper).** 

## Dependencies:
We develop the codes on Windows operation system, and run the codes on Ubuntu 22.04. The runtime environment for the code is the same as that of [BLoB](https://github.com/Wang-ML-Lab/bayesian-peft). 

## Dataset:
You can find dataset files at: `bayesian_peft/database` .

## Usage:

### 1. Estimate the target LLM uncertainty based on the BLoB project.
To accomplish our goal, we developed our code based on [BLoB](https://github.com/Wang-ML-Lab/bayesian-peft).
    
     cd bayesian_peft

Explain the contents of `scripts/uncertainty_estiamtion.sh`.

Fine-tuning MLL baseline:

     dr=0
     modelwrapper=mle

	for epoch in 3
	do
    		CUDA_VISIBLE_DEVICES=$devices python run/main.py --dataset-type mcdataset --dataset $train_dataset \
    		--model-type causallm --modelwrapper $modelwrapper  --lr $lr --batch-size $batchsize --opt adamw --warmup-ratio 0.1  \
    		--max-seq-len $maxlength \
     		--nowand  \
     		--lora-r 8 --lora-alpha 16  --model $llm   \
    		--load-model-path $project_path/my_llm/$llm --lora-dropout $dr  --checkpoint --seed $seed \
     		--n-epochs $epoch  --checkpoint-dic-name epoch$epoch/$model-1
	done
     
Inference:

     	for dataset in  vulcure_vd_${data_type}_test_800 vulcure_vd_${data_type}_correct_800
	do
	for epoch in 3
	do
    	CUDA_VISIBLE_DEVICES=$devices python run/main.py --dataset-type mcdataset --dataset $train_dataset \
    	--ood-ori-dataset $dataset --model-type causallm --model $llm --modelwrapper $modelwrapper \
    	--batch-size $batchsize  --max-seq-len $maxlength  --seed $seed \
    	--lora-dropout $dr  \
    	--nowand --load-model-path $project_path/my_llm/$llm \
    	--bayes-eval-index 1 --n-epochs $epoch \
    	--load-lora-path  $project_path/bayesian_peft/checkpoints/causallm_$modelwrapper/$llm/$train_dataset/epoch$epoch/$model-1
	done
	done

Hyperparameters:

     modelwrapper: uncertainty estimation method.
               'mcdropout': MCD, 'blob': BLoB, 'deepensemble': Ensemble

### 2. MalCertain.
     cd amyexperiments/uncertainty_malcertain
     python3 uncertainty_metrics2csv.py  ## generate a uncertainty feature csv file to amyexperiments/uncertainty_malcertain/feature_csv


     python3 malcertain.py

### 3. MalTutor.
     cd amyexperiments/uncertainty_maltutor
     python3 uncertainty_metrics2csv.py  ## generate a uncertainty feature csv file to amyexperiments/uncertainty_malcertain/feature_csv


     python3 dataset_reconstruction.py    ## generate train json file to bayesian_peft/database

     
     #### train MLL LLM
	bash scripts/maltutor.sh

     
     python3 evaluate_maltutor.py


