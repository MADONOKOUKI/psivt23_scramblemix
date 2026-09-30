# psivt23_scramblemix


- Experimental codes are stored under the `script` directory
- more details will be listed after the psivt 2023 proceedings
- we can use bash script to train the model, for example of bash script
```
num_of_TTA=2
js_diverfence=True
python main.py --model_name resnet18 \
               --dataset cifar10 \
               --save_directory_name resnet18_cifar10_"$num_of_TTA"_"$js_diverfence" \
               --milestones '100,150' \
               --weight_decay 5e-4 \
               --momentum 0.9 \
               --e 200\
               --num_of_TTA "$num_of_TTA" \
               --js_divergence_regularization "$js_diverfence" \
               --training_model_name random_pe_resnet18_cifar10_"$num_of_TTA"_"$js_diverfence".t7 \
               > main_resnet_cifar10_"$num_of_TTA"_"$js_diverfence".txt 
```