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
num_of_TTA=2
js_diverfence=True
python main.py --model_name resnet18 \
               --dataset cifar100 \
               --save_directory_name resnet18_cifar100_"$num_of_TTA"_"$js_diverfence" \
               --milestones '100,150' \
               --weight_decay 5e-4 \
               --momentum 0.9 \
               --e 200\
               --num_of_TTA "$num_of_TTA" \
               --js_divergence_regularization "$js_diverfence" \
               --training_model_name random_pe_resnet18_cifar100_"$num_of_TTA"_"$js_diverfence".t7 \
               > main_resnet_cifar100_"$num_of_TTA"_"$js_diverfence".txt 

