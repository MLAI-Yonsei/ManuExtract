
# **ManuExtract: Fine-Tuning LLaMa and Evaluation Workflow**

This document provides a comprehensive guide for fine-tuning and evaluating models using LLaMA Factory.  
<p align="center">
  <img width="623" src="https://github.com/user-attachments/assets/54a77711-4199-47bc-9c80-4dfc4ff3916a" alt="image">
</p>
The workflow includes cloning the repository, preprocessing data, registering datasets, running evaluations, and fine-tuning tasks.  
Follow these steps carefully to achieve optimal results.

---

## **1. Clone the LLaMA Factory Repository**
1. Clone the repository:
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory
   cd LLaMA-Factory
   ```

2. **Check Git Version**:
   - Navigate to the `LLaMA-Factory` folder and verify the HEAD commit:
     ```bash
     git log
     ```
   - The HEAD should match:
     ```
     commit 18e455c232d8c342d68195b118cceacd004ec368 (HEAD -> main, origin/main, origin/HEAD)
     ```
   - If the log points to a different commit, reset the repository:
     ```bash
     git reset --hard 18e455c232d8c342d68195b118cceacd004ec368
     ```

---

## **2. Data**
<p align="center">
  <img width="623" alt="image" src="https://github.com/user-attachments/assets/a6329511-db73-4629-8605-b1b231a8700e">
</p>
We have already uploaded preprocessed data.
   - Preprocessed data is available in  .

For fine-tuning, we utilized the data/train.json file contains a carefully designed dataset based on text augmentation approaches.   
The dataset was structured into multiple complementary tasks to improve the model’s ability to understand technical specifications from various perspectives.
<p align="center">
  <img width="623" alt="image" src="https://github.com/user-attachments/assets/42a4f3e9-84b9-4838-9f2f-2dcd231a54f4">
</p>
These tasks were designed to decompose the original information extraction problem into manageable subtasks, thereby creating a diverse training set.   
This approach enables the model to develop a deeper understanding of technical specifications.

---

## **3. Register the Dataset**
1. Place the JSON files generated in Step 2 under:
   ```
   ./LLaMA-Factory/data
   ```

2. Update `dataset_info.json` with the dataset details:
   ```json
   {
     "Data name": {
       "file_name": "Data Path"
     },
  
   ```

---

## **4. Evaluate the Dataset without FT model (ManuExtract) **
Run the evaluation with the following command:
```bash
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" CUDA_VISIBLE_DEVICES=7 python /LLaMA-Factory/src/train.py   --stage sft   --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct   --preprocessing_num_workers 16   --finetuning_type lora   --template llama3   --dataset_dir ./LLaMA-Factory/data   --eval_dataset "Data name"   --cutoff_len 1024   --max_samples 100000   --per_device_eval_batch_size 10   --predict_with_generate True   --max_new_tokens 512   --top_p 0.7   --temperature 0.001   --output_dir "Output path"   --do_predict True
```

### **Notes**:
- Adjust paths in `dataset_info.json`, `train.py`, and `--dataset_dir` as needed.
- Customize `--output_dir` to set the output location.
- Customize `--eval_dataset` to Evaluate your data.

---

## **5. Fine-Tuning**
### **Data Preparation**
- Prepare datasets for fine-tuning by following Step 2.

### **Fine-Tuning Parameters**
1. Install Required Packages:
   ```bash
   pip uninstall llmtuner
   pip install -e '.[metrics]'
   ```

2. Update YAML Configuration


3. Start Training:
   ```bash
   CUDA_VISIBLE_DEVICES=1,2,3,4 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
   ```

The fine-tuned checkpoint for ManuExtract and the corresponding YAML file used to generate it are available in the  `Model/` directory.  
These resources can be utilized for further fine-tuning or inference tasks.

---

### **Evaluate the Dataset with FT model (ManuExtract)**
Run the evaluation with the following command:
   ```bash
   NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" CUDA_VISIBLE_DEVICES=7 python /LLaMA-Factory/src/train.py --stage sft --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --preprocessing_num_workers 16 --finetuning_type lora --template llama3 --dataset_dir ./LLaMA-Factory/data --eval_dataset "Data name" --adapter_name_or_path "Your Checkpoint" --output_dir "Output path" --do_predict True
   ```

### **Notes**:
- Add `--adapter_name_or_path` to specify the fine-tuned checkpoint.

---

## **Acknowledgments**
This guide is designed to streamline your workflow with ManuExtract, enabling efficient fine-tuning and evaluation of models.   
Ensure all paths and configurations are aligned with your local setup for a seamless experience.
