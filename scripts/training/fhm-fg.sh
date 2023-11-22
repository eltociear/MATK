# # Fine-tunes Flava on Fine-Grain FHM

python3 main.py --multirun \
    +experiment=fhm_finegrained/flava.yaml \
    action=fit \
    seed_everything=1111,1112,1113,1114,1115,1116,1117,1118,1119,1120


# python3 main.py --multirun \
#     +experiment=fhm_finegrained/t5_classification.yaml \
#     action=fit \
#     datamodule.batch_size=16 \
#     trainer.accumulate_grad_batches=2 \
#     model.optimizers.0.lr=2e-5 \
#     seed_everything=1111,1112,1113,1114,1115,1116,1117,1118,1119,1120


# python3 main.py --multirun \
#     +experiment=fhm_finegrained/bart_classification.yaml \
#     action=fit \
#     datamodule.batch_size=16 \
#     model.optimizers.0.lr=2e-5 \
#     seed_everything=1111,1112,1113,1114,1115,1116,1117,1118,1119,1120
