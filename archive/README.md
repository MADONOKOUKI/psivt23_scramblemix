# Original research code (2023)

This folder holds the original research code used for the PSIVT 2023 paper
*ScrambleMix: A Privacy-Preserving Image Processing for Edge-Cloud Machine Learning*.
It is kept unchanged for reference and reproducibility and is **not maintained**: it targets
Python 3.6 with CUDA and `DataParallel`, uses relative `sys.path` hacks and cluster-specific paths,
and is not importable as a package. Use the `scramblemix` package and `train.py` at the
repository root instead; they re-implement the same pipeline and are tested against the files here
(`tests/test_scrambling.py`, `tests/test_transform.py`, `tests/test_losses.py`).

## Map from the old scripts to the new commands

The shell scripts `cd` into a cluster directory and set up a virtual environment before calling
`main.py`; only the training arguments matter. `num_of_TTA` is the number of ScrambleMix views per
training image (`--views` in the new code); the test-time augmentation always used four key pairs.

| Original script (in `scripts/scramblemix/`) | Settings in the script | New command |
|---|---|---|
| `main_paper.sh` | `senet2` (= ResNeXt-29 16x4d), CIFAR-100, num_of_TTA 4, JS on, 200 epochs, milestones 60/120/180 | `python train.py --dataset cifar100 --model resnext29_16x4d` |
| `main_shakedrop_cifar10.sh` | ShakeDrop, CIFAR-10, num_of_TTA 1, 200 epochs, 60/120/180 | `python train.py --dataset cifar10 --model shakedrop --views 1` |
| `main_shakedrop_cifar100.sh` | ShakeDrop, CIFAR-100, num_of_TTA 2, JS on, 300 epochs, 150/225 | `python train.py --dataset cifar100 --model shakedrop --views 2 --epochs 300 --milestones 150 225` |
| `main_shakedrop_cifar10_tta.sh` | ShakeDrop, CIFAR-10, num_of_TTA 3, 300 epochs, 150/225 | `python train.py --dataset cifar10 --model shakedrop --views 3 --epochs 300 --milestones 150 225` |
| `main_shakedrop_cifar100_tta.sh` | same for CIFAR-100 but `--e 1` (1 epoch) | `python train.py --dataset cifar100 --model shakedrop --views 3 --epochs 1` |
| `main_resnet18_cifar10.sh`, `main_resnet18_cifar100.sh` | `--model_name resnet18`, num_of_TTA 2, 200 epochs, 100/150 (builds ShakeDrop PyramidNet, see below) | `python train.py --dataset cifar10 --model shakedrop --views 2 --milestones 100 150` |
| `main.sh` | older launcher (`--num_of_keys`, 150 epochs, default milestones 75/125) | `python train.py --epochs 150 --milestones 75 125 --views 1` |

The paper's result tables (slides 39-40 of `../psivt.pdf`) are reproduced with the new defaults
(the `main_paper.sh` setting): `bash scripts/reproduce_accuracy.sh` at the repository root.

| Original file | Re-implemented in |
|---|---|
| `scripts/scramblemix/cifar10.py` (`AugMixDataset`, eight hard-coded keys per dataset) | `scramblemix.ScrambleMix`, `ScrambleMixViews`, `EvalViews`; keys in `scramblemix/resources/original_pe_keys.json` (`ScrambleMix.original(dataset)`) |
| `scripts/scramblemix/pixel_based_encryption.py` | `scramblemix.PixelEncryption` (`channel_mode="original"`) |
| `scripts/scramblemix/learnable_encryption_augmix.py`, `utils/learnable_encryption.py`, `*/Blockwise_scramble_LE.py`, `utils/key4/*.pkl` | `scramblemix.LearnableEncryption` (`LearnableEncryption.from_pickle("archive/utils/key4/0_.pkl")`, `train.py --scheme le`) |
| `scripts/scramblemix/trainer.py` (`train`: CE + JS term; `test`: single view, 4-key TTA, LPIPS) | `scramblemix.scramblemix_loss`, `self_teaching_loss`, `average_predictions`, `predict_tta`, `lpips_distance`; `train.py`, `evaluate_lpips.py` |
| `scripts/scramblemix/main.py`, `parameter.py`, `dataloader.py` | `train.py`, `scramblemix/data.py` |
| `models/no_adaptation_network.py`, `models/shakedrop.py` | `scramblemix.models.ShakePyramidNet` (`--model shakedrop`) |
| `scripts/scramblemix/wideresnet.py` | `scramblemix.models.WideResNet` (`--model wrn40_2`) |
| `scripts/scramblemix/resnext2.py` | `scramblemix.models.ResNeXt` (`--model resnext29_16x4d`) |

Not used by the ScrambleMix pipeline and not re-implemented: `densenet.py`, `resnext.py` (SENet-18,
`--model_name senet`), `resnet.py`, `vgg.py`, `wide_resnet.py`, `batchnorm.py` and the
`Synchronized_BatchNorm_PyTorch` entry (a dangling git submodule pointer to commit `5768ead`,
presumably of https://github.com/vacancy/Synchronized-BatchNorm-PyTorch; no `.gitmodules` was
committed), `mixup.py`, `utils/mixup.py`, `utils/scheduler.py` (CyclicLR),
`models/no_adaptation_network_multitask.py`, `scramble_parameter.py` (alternative key sets; contains a
syntax error and is never imported) and `results.json` (empty).

## Behaviour of the original code worth knowing

These points were found while re-implementing and testing the code; the new implementation follows
the original behaviour where it affects the results and documents the rest.

1. **Keys.** `cifar10.py` hard-codes eight pixel-based encryption keys per dataset and pairs them as
   (0, 1), (2, 3), (4, 5), (6, 7). Training uses the first `num_of_TTA` pairs (one view each), the
   single-view test draws one of those pairs at random, and the test-time augmentation always averages
   all four pairs. The mixing ratio is drawn for every view with `alpha = 5e-3`, so `m` is nearly
   always close to 0 or 1.
2. **Channel shuffle.** `pixel_based_encryption.py` permutes the colour channels with
   `img0[:,:,0], img0[:,:,1], img0[:,:,2] = img0[:,:,p0], img0[:,:,p1], img0[:,:,p2]`. The right-hand side
   holds numpy views, so only the identity code keeps all three channels; the other five codes
   duplicate a channel. `PixelEncryption(channel_mode="original")` reproduces this exactly (it produced
   the paper's numbers); `channel_mode="permutation"` gives the invertible textbook version.
3. **Value range.** The scrambled images reach the network in [0, 255] (`ToTensor` does not rescale
   float arrays) while the clean images are in [0, 1]. Every network used here starts with a
   convolution followed by batch normalisation, so the input scale does not change what is learned; the
   new code uses [0, 1]. The LPIPS score printed by `trainer.py` compares a [0, 1] image with a [0, 255]
   image without normalisation; `evaluate_lpips.py` feeds both in [0, 1] (`normalize=True`).
4. **Self-teaching loss.** `trainer.py` adds the generalised Jensen-Shannon term
   `1/D sum_d KL(p_d || mean_d p_d)` with weight 1 and no stop-gradient; the slides write it with a
   stop-gradient on the average posterior. Value and gradients are identical (tested).
5. **Boolean flag.** `--js_divergence_regularization` is declared with `type=bool`, so any value given
   on the command line, including `False`, enables the term. With `num_of_TTA 1` the term is zero anyway.
6. **Model names.** `main.py` builds ShakePyramidNet (PyramidNet-110, alpha 270, ShakeDrop) for every
   `--model_name` other than `densenet`, `wideresnet`, `senet` and `senet2`, so the `main_resnet18_*.sh`
   runs trained ShakePyramidNet. `wideresnet` is `WideResNet(num_classes=...)` with the file's defaults
   `depth=40, widen_factor=2` (the slides label the network "WideResNet40x10").
7. **TTA.** The test-time augmentation averages logits (the slides average posteriors); see
   `--tta-reduction` in the new `train.py`.
8. **Optimiser and schedule.** The batch size is hard-coded to 256 in `dataloader.py` (`--batch_size` is
   unused); `--nesterov` is never passed to SGD. `scheduler.step()` is called at the start of every
   epoch, which with PyTorch >= 1.1 decays the learning rate one epoch before each milestone; the new
   code steps the scheduler after each epoch.
9. **Unused forward passes.** All four training views are forwarded even when `num_of_TTA < 4`; only the
   first `num_of_TTA` enter the loss (the extra passes only update BatchNorm statistics). The new code
   forwards only the views it uses.
10. **Model selection.** The checkpoint is saved whenever the single-view test accuracy improves and the
    last epoch is evaluated ten times; `train.py` reports the mean of `--final-repeats 10` evaluations of
    the final model and also records the best per-epoch single-view accuracy.

## Original README (2023)

> - Experimental codes are stored under the `script` directory
> - more details will be listed after the psivt 2023 proceedings
> - we can use bash script to train the model, for example of bash script
>
> ```
> num_of_TTA=2
> js_diverfence=True
> python main.py --model_name resnet18 \
>                --dataset cifar10 \
>                --save_directory_name resnet18_cifar10_"$num_of_TTA"_"$js_diverfence" \
>                --milestones '100,150' \
>                --weight_decay 5e-4 \
>                --momentum 0.9 \
>                --e 200\
>                --num_of_TTA "$num_of_TTA" \
>                --js_divergence_regularization "$js_diverfence" \
>                --training_model_name random_pe_resnet18_cifar10_"$num_of_TTA"_"$js_diverfence".t7 \
>                > main_resnet_cifar10_"$num_of_TTA"_"$js_diverfence".txt
> ```
