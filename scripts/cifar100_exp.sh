for seed in 0 1 2
do
  for beta in 0.5 0.1 0.01
  do
    for alg in fedavg scaffold fednova moon
    do
    python experiments.py --model=simple-cnn \
      --dataset=cifar100 \
      --alg=$alg \
      --lr=0.01 \
      --batch-size=64 \
      --epochs=10 \
      --n_parties=10 \
      --rho=0.9 \
      --comm_round=20 \
      --partition=noniid-labeldir \
      --beta=$beta\
      --device='cuda:0'\
      --datadir='./data/' \
      --logdir='./logs/' \
      --noise=0\
      --sample=1\
      --init_seed=$seed
    done

    for mu in 0.001 0.01 0.1 1
    do
    python experiments.py --model=simple-cnn \
      --dataset=cifar100 \
      --alg=fedprox \
      --lr=0.01 \
      --batch-size=64 \
      --epochs=10 \
      --n_parties=10 \
      --rho=0.9 \
      --mu=$mu
      --comm_round=20 \
      --partition=noniid-labeldir \
      --beta=$beta\
      --device='cuda:0'\
      --datadir='./data/' \
      --logdir='./logs/' \
      --noise=0\
      --sample=1\
      --init_seed=$seed
    done
  done
done