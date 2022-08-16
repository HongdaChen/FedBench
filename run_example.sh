python experiments.py --model=ConvNet \
	--dataset=cifar10 \
	--alg=fedmeta \
	--lr=0.01 \
	--batch_size=64 \
	--epochs=10 \
	--n_parties=10 \
	--rho=0.9 \
	--comm_round=100 \
	--partition=noniid-labeldir \
	--beta=0.5 \
	--datadir='data' \
	--logdir='logs' \
	--noise=0 \
	--init_seed=1