"""
@Desc:
@Reference:
@Notes:

Bart uses the eos_token_id as the starting token for decoder_input_ids generation.
If past_key_values is used, optionally only the last decoder_input_ids have to be input (see past_key_values).
For translation and summarization training, decoder_input_ids should be provided. If no decoder_input_ids is provided,
the model will create this tensor by shifting the input_ids to the right for denoising pre-training following the paper.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from transformers import BartTokenizer
from transformers.models.bart import modeling_bart
from transformers.models.bart.modeling_bart import (
    BartConfig
)

from src.modules.event_trigger.datasets import (
    LeadingEventDataset,
    LeadingEventSbertDataset,
    LeadingSbertDataset,
    EventSbertDataset,
)
from src.modules.event_trigger.event_trigger_modules import EventBartForCG
from src.utils.event_trigger import model_utils
from src.utils.file_utils import save_json
from src.models.event_trigger.event_bart import EventBart, LeadingContextBart
from src.utils.gen_utils import top_p_logits
from src.utils import nlg_eval_utils
from src.configuration.constants import BASE_DIR

logger = logging.getLogger(__name__)


class EventLM(EventBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: BartConfig = BartConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model: EventBartForCG = self._load_model(self.hparams.model_name_or_path, EventBartForCG, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = LeadingEventDataset

        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix

    def init_for_vanilla_weights(self):
        print("================ init for vanilla weights =====================")
        model_path = self.hparams.model_name_or_path
        if ("bart-base" in model_path) or ("leading" in model_path) or ("event" in model_path):
            print(f"load parameters from {self.hparams.model_name_or_path}")
            print("clone the weights of event encoder from the encoder of vanilla pre-trained hugging face bart-base.")
            self.model.clone_weights()

    def save_readable_batch_fn(self, batch: Dict) -> Dict:
        """A debugging utility"""
        readable_batch = {}
        readable_batch["leading_contexts"] = {
            key: self.tokenizer.batch_decode(val.tolist()) for key, val in batch["leading_contexts"].items()
        }
        readable_batch["event_lines"] = {
            key: self.tokenizer.batch_decode(val.tolist()) for key, val in batch["leading_contexts"].items()
        }
        save_json(readable_batch, Path(self.experiment_output_dir) / "text_batch.json")

        self.already_saved_batch = True
        return readable_batch

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def _step(self, batch: dict):
        leading_contexts = batch["leading_contexts"]
        event_lines = batch["event_lines"]
        tgt_ids = batch["labels"]

        pad_token_id = self.tokenizer.pad_token_id
        decoder_input_ids = modeling_bart.shift_tokens_right(tgt_ids,
                                                             pad_token_id,
                                                             self.decoder_start_token_id)
        if self.save_readable_batch and not self.already_saved_batch:
            # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch_fn(batch)
        outputs = self(leading_input_ids=leading_contexts["input_ids"],
                       leading_attention_mask=leading_contexts["attention_mask"],
                       event_input_ids=event_lines["input_ids"],
                       event_attention_mask=event_lines["attention_mask"],
                       decoder_input_ids=decoder_input_ids,
                       use_cache=False,
                       output_attentions=True, output_hidden_states=True)

        lm_logits = outputs["logits"]

        if self.hparams.label_smoothing == 0:
            assert lm_logits.shape[-1] == self.vocab_size
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
            # lm_ligits: [batch, seq, vocab] tgt_ids: [batch, seq]
            losses_ = self.loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
            loss = torch.mean(losses_)
        else:
            lprobs = torch.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = model_utils.label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        lm_loss = loss
        return lm_loss

    @torch.no_grad()
    def _generative_step(self, batch: dict, fast_generate=False) -> dict:
        if fast_generate:
            print(f"fast_generate is not supported for {self.model_name}")
        return super()._generative_step(batch, fast_generate=False)

    def training_step(self, batch, batch_idx) -> Dict:
        loss = self._step(batch)
        logs = {"loss": loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        # tokens per batch
        logs["batch_size"] = batch["leading_contexts"]["input_ids"].shape[0]
        return {"loss": loss, "log": logs}

    @torch.no_grad()
    def sample_sequence(self, batch, use_top_p=False, top_p=0.9):
        leading_contexts = batch["leading_contexts"]
        event_lines = batch["event_lines"]
        batch_size = len(batch["ids"])
        decoder_input_ids = torch.tensor([self.tokenizer.eos_token_id for _
                                          in range(batch_size)])[:, None].to(self.device)
        for _ in range(self.hparams.max_target_length):
            outputs = self(leading_input_ids=leading_contexts["input_ids"],
                           leading_attention_mask=leading_contexts["attention_mask"],
                           event_input_ids=event_lines["input_ids"],
                           event_attention_mask=event_lines["attention_mask"],
                           decoder_input_ids=decoder_input_ids,
                           use_cache=False, return_dict=True)
            logits = outputs["logits"]
            logits = logits[:, -1, :]
            if use_top_p:
                logits = top_p_logits(logits, p=top_p, device=self.device)
                probs = torch.softmax(logits, dim=-1)
                pred = torch.multinomial(probs, 1)
            else:
                probs = torch.softmax(logits, dim=-1)
                pred = torch.topk(input=probs, k=1).indices
            decoder_input_ids = torch.cat([decoder_input_ids, pred], 1)
            # early stop
            if pred[:, 0].eq(self.tokenizer.eos_token_id).sum() == pred.shape[0]:
                break
        generated_ids = decoder_input_ids
        return generated_ids

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> LeadingEventDataset:
        event_infix = ""
        if "train" in src_file_prefix:
            event_infix = self.train_event_infix
        elif "val" in src_file_prefix:
            event_infix = self.eval_event_infix
        elif "test" in src_file_prefix:
            event_infix = self.test_event_infix
        else:
            NotImplementedError()
        dataset = self.dataset_class(
            self.tokenizer,
            src_file_prefix=src_file_prefix,
            event_file_prefix=f"{src_file_prefix}"
                              f"{event_infix}",
            tgt_file_prefix=tgt_file_prefix,
            max_target_length=self.hparams.max_target_length,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
        )
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

    def train_dataloader(self) -> DataLoader:
        train_shuffle = True if self.hparams.overfit_batches == 0.0 else False
        if not train_shuffle:
            print(f"train_shuffle: {train_shuffle} overfit_batches: {self.hparams.overfit_batches}")
        return self.get_dataloader("train", "train", batch_size=self.hparams.train_batch_size, shuffle=train_shuffle)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", "val", batch_size=self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", "test", batch_size=self.hparams.eval_batch_size, shuffle=False)

class EventLMSbert(EventLM):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.sbert_switch = True

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = LeadingEventSbertDataset

        self.sbert_linear_layer = torch.nn.Linear(self.config.d_model, self.config.d_model, bias=True)

    def _step(self, batch: dict):
        leading_contexts = batch["leading_contexts"]
        event_lines = batch["event_lines"]
        tgt_ids = batch["labels"]

        pad_token_id = self.tokenizer.pad_token_id
        decoder_input_ids = modeling_bart.shift_tokens_right(tgt_ids,
                                                             pad_token_id,
                                                             self.decoder_start_token_id)
        if self.save_readable_batch and not self.already_saved_batch:
            # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch_fn(batch)
        outputs = self(leading_input_ids=leading_contexts["input_ids"],
                       leading_attention_mask=leading_contexts["attention_mask"],
                       event_input_ids=event_lines["input_ids"],
                       event_attention_mask=event_lines["attention_mask"],
                       decoder_input_ids=decoder_input_ids,
                       use_cache=False,
                       output_attentions=True, output_hidden_states=True)

        lm_logits = outputs["logits"]

        if self.hparams.label_smoothing == 0:
            assert lm_logits.shape[-1] == self.vocab_size
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
            # lm_ligits: [batch, seq, vocab] tgt_ids: [batch, seq]
            losses_ = self.loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
            loss = torch.mean(losses_)
        else:
            lprobs = torch.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = model_utils.label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        lm_loss = loss + 0.
        # sbert loss
        sbert_loss = torch.tensor(0., )

        if not self.sbert_switch:
            return (loss, lm_loss, sbert_loss)

        # [batch_size, sequence_length, hidden_size]
        hidden_states = outputs["decoder_hidden_states"][-1]
        batch_size, sequence_length, hidden_size = hidden_states.size()
        # [batch_size, sequence_length]
        sen_pos = batch["labels"].eq(self.tokenizer.mask_token_id).to(torch.float)

        try:
            sen_idx = sen_pos.nonzero()

            # [bath_size, sentence_num, sentence_num]
            sbert_score_label = batch["sbert_score"]
            sentence_num = sbert_score_label.size()[1]

            # [batch_size, sentence_num, hidden_size]
            sent_hidden_states_gather = self.gather_nd(hidden_states, sen_idx)
            sent_hidden_states = torch.reshape(sent_hidden_states_gather, [batch_size, sentence_num, hidden_size])
            self.sbert_linear_layer = self.sbert_linear_layer.float()

            # [batch_size, sentence_num, sentence_num]
            pred = torch.matmul(self.sbert_linear_layer(sent_hidden_states), torch.transpose(sent_hidden_states, 1, 2))
            pred_score = -1 + 2 * torch.sigmoid(pred + torch.transpose(pred, 1, 2))

            sbert_mask = torch.ones_like(pred_score)

            batch_sbert_loss = torch.max(torch.abs(pred_score - sbert_score_label) - 0.1,
                                         torch.zeros_like(pred_score).to(pred_score.device))
            batch_sbert_loss *= sbert_mask
            sbert_loss = 0.1 * torch.sum(batch_sbert_loss) / (torch.sum(sbert_mask) + 1e-20)
        except Exception as e:
            # there is a bug in the test process
            if self.training:
                raise e
            else:
                print("error occured when calculating sbert_loss.")
                print(str(e))
                try:
                    print(f"the shape of sent_hidden_states_gather: {sent_hidden_states_gather.shape}")
                except Exception:
                    pass

        loss += sbert_loss
        return (loss, lm_loss, sbert_loss)

    def training_step(self, batch, batch_idx) -> Dict:
        loss, lm_loss, sbert_loss = self._step(batch)
        logs = {"loss": loss.item(), "lm_loss": lm_loss.item(), "sbert_loss": sbert_loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        # tokens per batch
        logs["batch_size"] = batch["leading_contexts"]["input_ids"].shape[0]
        return {"loss": loss, "log": logs}

    @torch.no_grad()
    def _generative_step(self, batch: dict, fast_generate=False) -> dict:
        tik = datetime.now()
        if fast_generate:
            print(f"fast_generate is not supported for {self.model_name}")
        generated_ids = self.sample_sequence(batch, use_top_p=self.use_top_p, top_p=self.top_p)
        tok = datetime.now()
        batch_gen_time = tok - tik
        preds: List[str] = self.gen_ids_to_clean_text(generated_ids)
        targets: List[str] = self.gen_ids_to_clean_text(batch["labels"])
        loss, lm_loss, sbert_loss = self._step(batch)

        base_metrics = {"loss": loss.item(), "lm_loss": lm_loss.item(), "sbert_loss": sbert_loss.item()}
        rouge_metrics: Dict = nlg_eval_utils.calculate_rouge(pred_lines=preds, tgt_lines=targets)
        base_metrics.update(**rouge_metrics)
        bleu_metrics: Dict = nlg_eval_utils.calculate_bleu(ref_lines=[self.tokenizer.tokenize(l) for l in targets],
                                                           gen_lines=[self.tokenizer.tokenize(l) for l in preds])
        base_metrics.update(**bleu_metrics)
        summ_len = np.mean(list(map(len, generated_ids)))

        # update metric_names
        self.update_metric_names(base_metrics, update_flag=self.metric_names_update_flag)
        self.metric_names_update_flag = False
        base_metrics.update(batch_gen_time=batch_gen_time, gen_len=summ_len,
                            preds=preds, targets=targets)
        return base_metrics

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> LeadingEventSbertDataset:
        event_infix = ""
        if "train" in src_file_prefix:
            event_infix = self.train_event_infix
        elif "val" in src_file_prefix:
            event_infix = self.eval_event_infix
        elif "test" in src_file_prefix:
            event_infix = self.test_event_infix
        else:
            NotImplementedError()
        datadir_name_ = Path(self.hparams.data_dir).name
        dataset = self.dataset_class(
            self.tokenizer,
            src_file_prefix=src_file_prefix,
            event_file_prefix=f"{src_file_prefix}"
                              f"{event_infix}",
            tgt_file_prefix=tgt_file_prefix,
            max_target_length=self.hparams.max_target_length,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            sbert_data_dir=f"{BASE_DIR}/datasets/thu-coai-hint/{datadir_name_}"
        )
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset
