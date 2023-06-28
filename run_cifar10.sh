# EXPERIMENTS WITH MATRIX MULTIPLICATION (MSCoSSL)

# ReMixMatch-with-maximum-separation@r=5
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r5_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r5_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r5_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True

# ReMixMatch-with-maximum-separation@r=10
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r10_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r10_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r10_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True

# ReMixMatch-with-maximum-separation@r=50
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r50_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r50_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r50_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True

# ReMixMatch-with-maximum-separation@r=100
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r100_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r100_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r100_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True

# ReMixMatch-with-maximum-separation@r=150
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r150_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r150_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True

# ReMixMatch-with-maximum-separation@r=400
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r400_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r400_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r400_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True


# MixMatch-with-maximum-separation@r=5
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r5_seed1/ --manualSeed 1 --gpu 0 --matrix True
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r5_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r5_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# MixMatch-with-maximum-separation@r=10
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r10_seed1/ --manualSeed 1 --gpu 0 --matrix True
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r10_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r10_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# MixMatch-with-maximum-separation@r=50
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r50_seed1/ --manualSeed 1 --gpu 0 --matrix True
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r50_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r50_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# MixMatch-with-maximum-separation@r=100
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r100_seed1/ --manualSeed 1 --gpu 0 --matrix True
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r100_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r100_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# MixMatch-with-maximum-separation@r=150
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r150_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r150_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# MixMatch-with-maximum-separation@r=400
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r400_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r400_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r400_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True


# FixMatch-with-maximum-separation@r=5
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r5_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r5_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r5_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# FixMatch-with-maximum-separation@r=10
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r10_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r10_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r10_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# FixMatch-with-maximum-separation@r=50
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r50_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r50_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r50_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# FixMatch-with-maximum-separation@r=100
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r100_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r100_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r100_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# FixMatch-with-maximum-separation@r=150
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r150_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r150_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# FixMatch-with-maximum-separation@r=400
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r400_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r400_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r400_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# EXPERIMENTS WITHOUT MATRIX MULTIPLICATION (ORIGINAL CoSSL)

# ReMixMatch@r=5
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r5_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r5_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r5_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=10
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r10_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r10_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r10_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=50
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r50_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r50_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r50_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=100
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r100_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r100_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r100_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=150
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r150_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r150_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=400
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r400_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r400_seed1/checkpoint.pth.tar --out ./results/cifar10/remixmatch/mscossl/wrn28_N1500_r400_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0


# MixMatch@r=5
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r5_seed1/ --manualSeed 1 --gpu 0
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r5_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r5_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# MixMatch@r=10
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r10_seed1/ --manualSeed 1 --gpu 0
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r10_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r10_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# MixMatch@r=50
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r50_seed1/ --manualSeed 1 --gpu 0
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r50_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r50_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# MixMatch@r=100
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r100_seed1/ --manualSeed 1 --gpu 0
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r100_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r100_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# MixMatch@r=150
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 0
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r150_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r150_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# MixMatch@r=400
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r400_seed1 --manualSeed 1 --gpu 0
python train_cifar_mix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r400_seed1/checkpoint.pth.tar --out ./results/cifar10/mixmatch/mscossl/wrn28_N1500_r400_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0



# FixMatch@r=5
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r5_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r5_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r5_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=10
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r10_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r10_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r10_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=50
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r50_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r50_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r50_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=100
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r100_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r100_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r100_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=150
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r150_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r150_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=400
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r400_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --ratio 2 --num_max 1500 --imb_ratio_l 400 --imb_ratio_u 400 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r400_seed1/checkpoint.pth.tar --out ./results/cifar10/fixmatch/mscossl/wrn28_N1500_r400_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0
