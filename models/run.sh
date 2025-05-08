#!/usr/bin/env bash
#SBATCH --job-name=UNBERT       # Job name
#SBATCH --output=bert2/UNBERT-%j.out  # Standard output and error log
#SBATCH --mail-user=jcollier@middlebury.edu
#SBATCH --mail-type=END,FAIL            # Mail events
#SBATCH --mem=50GB                      # Job memory request CHANGE
#SBATCH --partition=gpu-long            # Partition (queue) / or do long
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                    # Request GPU / can be up to 4

# activate environment
# conda activate bert_env_2

# install packages
#conda install -y pandas scikit-learn argparse pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch transformers datasets evaluate -c anaconda xlrd
#pip install datetime

# show which python we’re using
which python

cd $HOME/bert2

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Starting: $(date +"%D %T")"

# TRAINING
#python classification_train3_1.py --mode train --data_file cleaned_data_for_classifying_annotation.csv --output_dir warmed_actor_tune

# PREDICTION
python actor_classification.py --mode predict --data_file FULL_UNResolutionData_2024_09_12_0534PM.csv --model_dir warmed_actor_tune --output_file 4_30_predictions.csv

# PREDICTION (add sentiment)
python zeroshot_sentiment_prediction.py --predictions_csv 4_30_predictions.csv --output_csv 4_30_predictions_with_sentiments.csv

echo "Ending: $(date +"%D %T")"