for seed in 0 1 2
do
	for beta in 0.5 0.1 0.01
	do
		for alg in fedavg
		do
			python ../experiments.py --model=ConvNet \
				--dataset=cifar100 \
				--alg=$alg \
				--lr=0.01 \
				--batch_size=64 \
				--epochs=10 \
				--n_parties=10 \
				--rho=0.9 \
				--comm_round=20 \
				--partition=noniid-labeldir \
				--beta=$beta \
				--device='cuda:0' \
				--datadir='../data/' \
				--logdir='../logs/' \
				--noise=0 \
				--sample=1 \
				--init_seed=$seed
		done
	done
done
