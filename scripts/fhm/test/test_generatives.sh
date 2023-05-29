python3 main.py fit \
    --config configs/fhm/hate_cls/bart.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 3 \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2


python3 main.py fit \
    --config configs/fhm/hate_cls/t5.yaml \
    --seed_everything 1111 \
    --trainer.devices 1 \
    --trainer.max_epochs 3  \
    --trainer.limit_train_batches 5 \
    --trainer.limit_val_batches 2