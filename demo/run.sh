mkdir ../data/temp
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python semeval_demo.py --make_data --train --predict
