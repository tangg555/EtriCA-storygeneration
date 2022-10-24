#!/bin/bash
set -e

# =============================== train ====================
# leading -------------------------------
python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name leading-bart --experiment_name=leading-bart-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name leading-hint --experiment_name=leading-hint-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name leading-sbert-bart --experiment_name=leading-sbert-bart-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

# event -------------------------------
python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name event-bart --experiment_name=event-bart-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/writing-prompts\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name event-bart --experiment_name=event-bart-writing-prompts\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/writing-prompts\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name event-bart --experiment_name=event-bart-writing-prompts\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name event-hint --experiment_name=event-hint-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name event-sbert-bart --experiment_name=event-sbert-bart-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/writing-prompts\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name event-sbert-bart --experiment_name=event-sbert-bart-writing-prompts\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

# leading plus event -------------------------------
python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=8e-5 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/writing-prompts\
 --learning_rate=8e-5 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-writing-prompts\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name leading-plus-event-hint --experiment_name=leading-plus-event-hint-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-plus-event-bart-roc-stories/best_tfmr \
 --output_dir=output/event-trigger --model_name event-lm --experiment_name=event-lm-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-lm-roc-stories/best_tfmr \
 --output_dir=output/event-trigger --model_name event-lm-sbert --experiment_name=event-lm-sbert-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=8e-5  --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base  \
 --output_dir=output/event-trigger --model_name event-lm-sbert-no-cm --experiment_name=event-lm-sbert-no-cm-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/writing-prompts\
 --learning_rate=1e-4  --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base  \
 --output_dir=output/event-trigger --model_name event-lm-sbert-no-cm --experiment_name=event-lm-sbert-no-cm-writing-prompts\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0
## leading-to-events ----------------------
python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=15 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name leading-to-events-bart --experiment_name=leading-to-events-bart-roc-stories  \
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4 --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=15 --model_name_or_path=output/event-trigger/leading-bart-roc-stories/best_tfmr \
 --output_dir=output/event-trigger --model_name leading-to-events-bart --experiment_name=leading-to-events-bart-roc-stories  \
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4 --num_sanity_val_steps=0

python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/writing-prompts\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=15 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name leading-to-events-bart --experiment_name=leading-to-events-bart-writing-prompts  \
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4 --num_sanity_val_steps=0
# =============================== test ====================
# leading -------------------------------
python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-bart-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name leading-bart --experiment_name=leading-bart-roc-stories\
  --test_event_infix=_predicted_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-hint-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name leading-hint --experiment_name=leading-hint-roc-stories\
  --test_event_infix=_predicted_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-sbert-bart-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name leading-sbert-bart --experiment_name=leading-sbert-bart-roc-stories\
  --test_event_infix=_event

# event -------------------------------
python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-bart-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name event-bart --experiment_name=event-bart-roc-stories\
  --test_event_infix=_predicted_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/writing-prompts\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-bart-writing-prompts/best_tfmr \
  --output_dir=output/event-trigger --model_name event-bart --experiment_name=event-bart-writing-prompts\
  --test_event_infix=_predicted_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-hint-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name event-hint --experiment_name=event-hint-roc-stories\
  --test_event_infix=_predicted_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-hint-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name event-hint --experiment_name=event-hint-roc-stories\
  --test_event_infix=_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/writing-prompts\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-hint-writing-prompts/best_tfmr \
  --output_dir=output/event-trigger --model_name event-hint --experiment_name=event-hint-writing-prompts\
  --test_event_infix=_predicted_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-sbert-bart-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name event-sbert-bart --experiment_name=event-sbert-bart-roc-stories\
  --test_event_infix=_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-sbert-bart-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name event-sbert-bart --experiment_name=event-sbert-bart-roc-stories\
  --test_event_infix=_predicted_event

# leading plus event -------------------------------
python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-plus-event-bart-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-roc-stories\
  --test_event_infix=_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/writing-prompts\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-plus-event-bart-writing-prompts/best_tfmr \
  --output_dir=output/event-trigger --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-writing-prompts\
  --test_event_infix=_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-plus-event-hint-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name leading-plus-event-hint --experiment_name=leading-plus-event-hint-roc-stories\
  --test_event_infix=_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-lm-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name event-lm --experiment_name=event-lm-roc-stories\
  --test_event_infix=_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-lm-sbert-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name event-lm-sbert --experiment_name=event-lm-sbert-roc-stories\
  --test_event_infix=_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-lm-sbert-no-cm-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name event-lm-sbert-no-cm --experiment_name=event-lm-sbert-no-cm-roc-stories\
  --test_event_infix=_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/writing-prompts\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-lm-sbert-no-cm-writing-prompts/best_tfmr \
  --output_dir=output/event-trigger --model_name event-lm-sbert-no-cm --experiment_name=event-lm-sbert-no-cm-writing-prompts\
  --test_event_infix=_event

# _predicted
python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-plus-event-bart-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-roc-stories\
  --test_event_infix=_predicted_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-plus-event-bart-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-roc-stories\
  --test_event_infix=_bart_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/writing-prompts\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-plus-event-bart-writing-prompts/best_tfmr \
  --output_dir=output/event-trigger --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-writing-prompts\
  --test_event_infix=_predicted_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-plus-event-hint-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name leading-plus-event-hint --experiment_name=leading-plus-event-hint-roc-stories\
  --test_event_infix=_predicted_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-lm-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name event-lm --experiment_name=event-lm-roc-stories\
  --test_event_infix=_predicted_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/writing-prompts\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-lm-writing-prompts/best_tfmr \
  --output_dir=output/event-trigger --model_name event-lm --experiment_name=event-lm-writing-prompts\
  --test_event_infix=_predicted_event

python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-lm-sbert-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name event-lm-sbert --experiment_name=event-lm-sbert-roc-stories\
  --test_event_infix=_predicted_event

## leading-to-events ----------------------
python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories \
 --eval_batch_size=15 --model_name_or_path=output/event-trigger/leading-to-events-bart-roc-stories/best_tfmr \
 --output_dir=output/event-trigger --model_name leading-to-events-bart --experiment_name=leading-to-events-bart-roc-stories \
 --test_event_infix=_event --remain_sp_tokens


