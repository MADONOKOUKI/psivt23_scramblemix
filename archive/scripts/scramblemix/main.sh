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
val=1
#python main.py --json_file_name main_"$val".json --num_of_keys="$val" --e=150 > main_reg_usual_soft_hard_label_learning.txt
#python main.py --json_file_name main_"$val".json --num_of_keys="$val" --e=150 > main_reg_usual_only_hard_label_learning.txt
python main.py --json_file_name main_"$val".json --num_of_keys="$val" --e=150 > scramble_mixup_shakedrop_1ce.txt #resnet18__2ce_js_alpha0.01_withjs.txt #2ce_js_minus1.txt #_0.01.txt #resnet_deepcopy_2ce_js.txt #shakepyramid.txt #wide_resnet.txt #main_reg_usual_only_hard_label_learning_withjs_2ce_lambda_1.txt #minus0.1.txt 
#python main.py --json_file_name main_"$val".json --num_of_keys="$val" --e=150 > main_reg_usual_only_hard_label_learning_js_rev.txt 
#python main.py --json_file_name main_"$val".json --num_of_keys="$val" --e=150 > main_reg_usual_only_soft_label_learning_js1_with_teacher_no_distillation_loss.txt
