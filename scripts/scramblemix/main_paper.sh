cd /groups1/gaa50073/madono/icip2021_classification
source ~/.bashrc
module load python/3.6/3.6.5
python3 -m venv work
source work/bin/activate
cd scripts/proposed_teacher_student_student_scramblemix_learning_kiya
module load python/3.6/3.6.5
module load cuda/10.1/10.1.243
module load cudnn/7.4/7.4.2
module load nccl/2.4/2.4.7-1
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
apt-get install python-dev
python3 -m pip install matplotlib
pip3 install torch torchvision tensorboardX
pip install -U scikit-learn
pip freeze > requirements.txt
num_of_TTA=4
js_diverfence=True
alpha=5e-3
pixelwise=False
classifier=senet2
dataset=cifar100
python main.py --model_name "$classifier" \
               --dataset "$dataset" \
               --save_directory_name "$classifier"_"$dataset"_"$num_of_TTA"_"$js_diverfence"_200epoch_mask_rev2_"$alpha"_paper_tta \
               --milestones '60,120,180' \
               --weight_decay 5e-4 \
               --momentum 0.9 \
               --e 200 \
               --num_of_TTA "$num_of_TTA" \
               --js_divergence_regularization "$js_diverfence" \
               --training_model_name random_pe_"$classifier"_"$dataset"_"$num_of_TTA"_"$js_diverfence"_200epoch_mask_rev2_"$alpha"_paper_tta.t7 \
               > bmvc_test_main_"$classifier"_"$dataset"_"$num_of_TTA"_"$js_diverfence"_200epoch_paper_rev2_"$alpha"_paper_tta.txt #mask_pixelwise.txt 

