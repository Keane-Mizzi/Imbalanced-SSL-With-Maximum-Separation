# EXPERIMENTS WITH MATRIX MULTIPLICATION (MSCoSSL)

# FixMatch-with-maximum-separation@r=5
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r5_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r5_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl-matrix/wrn28_N150_r5_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# FixMatch-with-maximum-separation@r=10
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r10_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r10_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl-matrix/wrn28_N150_r10_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# FixMatch-with-maximum-separation@r=20
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r20_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r20_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl-matrix/wrn28_N150_r20_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# FixMatch-with-maximum-separation@r=10
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r50_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r50_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl-matrix/wrn28_N150_r50_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# FixMatch-with-maximum-separation@r=100
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r100_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r100_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl-matrix/wrn28_N150_r100_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# FixMatch-with-maximum-separation@r=150
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r150_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline-matrix/wrn28_N150_r150_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl-matrix/wrn28_N150_r150_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0 --matrix True

# ReMixMatch-with-maximum-separation@r=5
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r5_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r5_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl-matrix/wrn28_N150_r5_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True

# ReMixMatch-with-maximum-separation@r=10
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r10_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r10_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl-matrix/wrn28_N150_r10_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True

# ReMixMatch-with-maximum-separation@r=20
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r20_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r20_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl-matrix/wrn28_N150_r20_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True

# ReMixMatch-with-maximum-separation@r=10
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r50_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r50_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl-matrix/wrn28_N150_r50_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True

# ReMixMatch-with-maximum-separation@r=100
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r100_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r100_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl-matrix/wrn28_N150_r100_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True

# ReMixMatch-with-maximum-separation@r=150
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r150_seed1 --manualSeed 1 --gpu 0 --matrix True
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline-matrix/wrn28_N150_r150_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl-matrix/wrn28_N150_r150_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0 --matrix True


# TESTS WITHOUT MATRIX MULTIPLICATION (ORIGINAL CoSSL)

# FixMatch@r=5
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline/wrn28_N150_r5_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline/wrn28_N150_r5_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl/wrn28_N150_r5_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=10
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline/wrn28_N150_r10_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline/wrn28_N150_r10_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl/wrn28_N150_r10_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=20
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline/wrn28_N150_r20_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline/wrn28_N150_r20_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl/wrn28_N150_r20_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=10
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline/wrn28_N150_r50_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline/wrn28_N150_r50_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl/wrn28_N150_r50_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=100
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline/wrn28_N150_r100_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline/wrn28_N150_r100_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl/wrn28_N150_r100_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=150
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/baseline/wrn28_N150_r150_seed1 --manualSeed 1 --gpu 0
python train_cifar_fix_mscossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline/wrn28_N150_r150_seed1/checkpoint.pth.tar --out ./results/cifar100/fixmatch/mscossl/wrn28_N150_r150_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0



# ReMixMatch@r=5
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline/wrn28_N150_r5_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 5 --imb_ratio_u 5 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline/wrn28_N150_r5_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl/wrn28_N150_r5_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=10
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline/wrn28_N150_r10_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 10 --imb_ratio_u 10 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline/wrn28_N150_r10_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl/wrn28_N150_r10_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=20
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline/wrn28_N150_r20_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline/wrn28_N150_r20_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl/wrn28_N150_r20_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=10
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline/wrn28_N150_r50_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline/wrn28_N150_r50_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl/wrn28_N150_r50_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=100
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline/wrn28_N150_r100_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline/wrn28_N150_r100_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl/wrn28_N150_r100_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=150
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/baseline/wrn28_N150_r150_seed1 --manualSeed 1 --gpu 0
python train_cifar_remix_mscossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline/wrn28_N150_r150_seed1/checkpoint.pth.tar --out ./results/cifar100/remixmatch/mscossl/wrn28_N150_r150_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0
