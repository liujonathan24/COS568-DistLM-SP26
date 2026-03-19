task 2a:
- python run_glue.py [other input args] --master_ip $ip_address$ --master_port $port$ --world_size 4 --local_rank $rank$

```
cd /users/jl0796/COS568-DistLM-SP26/
python download_glue_data.py
cd /users/jl0796/COS568-DistLM-SP26/task2a
export GLUE_DIR=$HOME/COS568-DistLM-SP26/glue_data
export TASK_NAME=RTE
export IP_ADDRESS=128.110.218.95
export PORT=13579
python3 run_glue.py   --model_type bert   --model_name_or_path bert-base-cased   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128   --per_device_train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 3   --output_dir /tmp/$TASK_NAME/   --overwrite_output_dir \
--master_ip $IP_ADDRESS --master_port $PORT --world_size 4 --local_rank [0-3]
```
master_ip is: 10.10.1.2
master_port: 13579


task 1:

- first 5 minibatches:

Step 0: 0.7691709399223328 loss                                                                                                                   | 0/39 [00:00<?, ?it/s]

Step 1: 0.7817338705062866 loss                                                                                                          | 1/39 [00:13<08:16, 13.07s/it]

Step 2: 0.6885838508605957 loss                                                                                                          | 2/39 [00:22<06:53, 11.17s/it]

Step 3: 0.7662752866744995 loss                                                                                                          | 3/39 [00:32<06:17, 10.47s/it]

Step 4: 0.7341869473457336 loss                                                                                                          | 4/39 [00:41<05:45,  9.89s/it]


- Full 3 epochs:
jl0796@node-0:~/COS568-DistLM-SP26$ python3 run_glue.py   --model_type bert   --model_name_or_path bert-base-cased   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128   --per_device_train_batch_size 64   --learning_rate 2e-5   --num_train_epochs 3   --output_dir /tmp/$TASK_NAME/   --overwrite_output_dir
03/15/2026 09:23:10 - WARNING - __main__ -   Process rank: -1, device: cpu, distributed training: False, 16-bits training: False
03/15/2026 09:23:10 - INFO - pytorch_transformers.modeling_utils -   loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json from cache at /users/jl0796/.cache/torch/pytorch_transformers/b945b69218e98b3e2c95acf911789741307dec43c698d35fad11c1ae28bda352.9da767be51e1327499df13488672789394e2ca38b877837e52618a67d7002391
03/15/2026 09:23:10 - INFO - pytorch_transformers.modeling_utils -   Model config {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": "rte",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_labels": 2,
  "output_attentions": false,
  "output_hidden_states": false,
  "pad_token_id": 0,
  "pruned_heads": {},
  "torchscript": false,
  "type_vocab_size": 2,
  "vocab_size": 28996
}

03/15/2026 09:23:10 - INFO - pytorch_transformers.tokenization_utils -   loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt from cache at /users/jl0796/.cache/torch/pytorch_transformers/5e8a2b4893d13790ed4150ca1906be5f7a03d6c4ddf62296c383f6db42814db2.e13dbb970cb325137104fb2e5f36fe865f27746c6b526f6352861b1980eb80b1
03/15/2026 09:23:11 - INFO - pytorch_transformers.modeling_utils -   loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json from cache at /users/jl0796/.cache/torch/pytorch_transformers/b945b69218e98b3e2c95acf911789741307dec43c698d35fad11c1ae28bda352.9da767be51e1327499df13488672789394e2ca38b877837e52618a67d7002391
03/15/2026 09:23:11 - INFO - pytorch_transformers.modeling_utils -   Model config {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_labels": 2,
  "output_attentions": false,
  "output_hidden_states": false,
  "pad_token_id": 0,
  "pruned_heads": {},
  "torchscript": false,
  "type_vocab_size": 2,
  "vocab_size": 28996
}

03/15/2026 09:23:11 - INFO - pytorch_transformers.modeling_utils -   loading weights file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin from cache at /users/jl0796/.cache/torch/pytorch_transformers/35d8b9d36faaf46728a0192d82bf7d00137490cd6074e8500778afed552a67e5.3fadbea36527ae472139fe84cddaa65454d7429f12d543d80bfc3ad70de55ac2
/users/jl0796/.local/lib/python3.10/site-packages/pytorch_transformers/modeling_utils.py:539: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state_dict = torch.load(resolved_archive_file, map_location='cpu')
03/15/2026 09:23:19 - INFO - pytorch_transformers.modeling_utils -   Weights of BertForSequenceClassification not initialized from pretrained model: ['classifier.weight', 'classifier.bias']
03/15/2026 09:23:19 - INFO - pytorch_transformers.modeling_utils -   Weights from pretrained model not used in BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
03/15/2026 09:23:19 - INFO - __main__ -   Training/evaluation parameters Namespace(data_dir='/users/jl0796/glue_data/RTE', model_type='bert', model_name_or_path='bert-base-cased', task_name='rte', output_dir='/tmp/RTE/', config_name='', tokenizer_name='', cache_dir='', max_seq_length=128, do_train=True, do_eval=True, do_lower_case=False, per_device_train_batch_size=64, per_device_eval_batch_size=8, gradient_accumulation_steps=1, learning_rate=2e-05, weight_decay=0.0, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=3.0, max_steps=-1, warmup_steps=0, no_cuda=False, overwrite_output_dir=True, overwrite_cache=False, seed=42, fp16=False, fp16_opt_level='O1', local_rank=-1, device=device(type='cpu'), n_gpu=0, output_mode='classification')
03/15/2026 09:23:19 - INFO - __main__ -   Loading features from cached file /users/jl0796/glue_data/RTE/cached_train_bert-base-cased_128_rte
/users/jl0796/COS568-DistLM-SP26/run_glue.py:244: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  features = torch.load(cached_features_file)
03/15/2026 09:23:21 - INFO - __main__ -   ***** Running training *****
03/15/2026 09:23:21 - INFO - __main__ -     Num examples = 2490
03/15/2026 09:23:21 - INFO - __main__ -     Num Epochs = 3
03/15/2026 09:23:21 - INFO - __main__ -     Instantaneous batch size per device = 64
03/15/2026 09:23:21 - INFO - __main__ -     Total train batch size (w. parallel, distributed & accumulation) = 64
03/15/2026 09:23:21 - INFO - __main__ -     Gradient Accumulation steps = 1
03/15/2026 09:23:21 - INFO - __main__ -     Total optimization steps = 117
Epoch:   0%|                                                                                                                                            | 0/3 [00:00<?, ?it/s/users/jl0796/.local/lib/python3.10/site-packages/pytorch_transformers/optimization.py:166: UserWarning: This overload of add_ is deprecated:           | 0/39 [00:00<?, ?it/s]
        add_(Number alpha, Tensor other)
Consider using one of the following signatures instead:
        add_(Tensor other, *, Number alpha = 1) (Triggered internally at ../torch/csrc/utils/python_arg_parser.cpp:1642.)
  exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
Iteration: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [05:36<00:00,  8.62s/it]
03/15/2026 09:28:57 - INFO - __main__ -   Creating features from dataset file at /users/jl0796/glue_data/RTE██████████████████████████████████| 39/39 [05:36<00:00,  8.02s/it]
03/15/2026 09:28:57 - INFO - utils_glue -   Writing example 0 of 277
03/15/2026 09:28:57 - INFO - utils_glue -   *** Example ***
03/15/2026 09:28:57 - INFO - utils_glue -   guid: dev-0
03/15/2026 09:28:57 - INFO - utils_glue -   tokens: [CLS] Dana Reeve , the widow of the actor Christopher Reeve , has died of lung cancer at age 44 , according to the Christopher Reeve Foundation . [SEP] Christopher Reeve had an accident . [SEP]
03/15/2026 09:28:57 - INFO - utils_glue -   input_ids: 101 11422 26034 117 1103 8244 1104 1103 2811 4978 26034 117 1144 1452 1104 13093 4182 1120 1425 3140 117 2452 1106 1103 4978 26034 2974 119 102 4978 26034 1125 1126 4216 119 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
03/15/2026 09:28:57 - INFO - utils_glue -   input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
03/15/2026 09:28:57 - INFO - utils_glue -   segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
03/15/2026 09:28:57 - INFO - utils_glue -   label: not_entailment (id = 1)
03/15/2026 09:28:57 - INFO - utils_glue -   *** Example ***
03/15/2026 09:28:57 - INFO - utils_glue -   guid: dev-1
03/15/2026 09:28:57 - INFO - utils_glue -   tokens: [CLS] Yet , we now are discovering that anti ##biotics are losing their effectiveness against illness . Disease - causing bacteria are m ##uta ##ting faster than we can come up with new anti ##biotics to fight the new variations . [SEP] Ba ##cter ##ia is winning the war against anti ##biotics . [SEP]
03/15/2026 09:28:57 - INFO - utils_glue -   input_ids: 101 6355 117 1195 1208 1132 15137 1115 2848 25523 1132 3196 1147 12949 1222 6946 119 20012 118 3989 10548 1132 182 15012 1916 4946 1190 1195 1169 1435 1146 1114 1207 2848 25523 1106 2147 1103 1207 9138 119 102 18757 25857 1465 1110 2183 1103 1594 1222 2848 25523 119 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
03/15/2026 09:28:57 - INFO - utils_glue -   input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
03/15/2026 09:28:57 - INFO - utils_glue -   segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
03/15/2026 09:28:57 - INFO - utils_glue -   label: entailment (id = 0)
03/15/2026 09:28:57 - INFO - utils_glue -   *** Example ***
03/15/2026 09:28:57 - INFO - utils_glue -   guid: dev-2
03/15/2026 09:28:57 - INFO - utils_glue -   tokens: [CLS] Cairo is now home to some 15 million people - a b ##urge ##oning population that produces approximately 10 , 000 tonnes of rub ##bish per day , putting an enormous strain on public services . In the past 10 years , the government has tried hard to encourage private investment in the refuse sector , but some estimate 4 , 000 tonnes of waste is left behind every day , f ##ester ##ing in the heat as it waits for someone to clear it up . It is often the people in the poor ##est neighbourhood ##s that are worst affected . But in some areas they are fighting back . In [SEP] 15 million tonnes of rub ##bish are produced daily in Cairo . [SEP]
03/15/2026 09:28:57 - INFO - utils_glue -   input_ids: 101 11086 1110 1208 1313 1106 1199 1405 1550 1234 118 170 171 27793 10087 1416 1115 6570 2324 1275 117 1288 10992 1104 16259 26652 1679 1285 117 4518 1126 7883 10512 1113 1470 1826 119 1130 1103 1763 1275 1201 117 1103 1433 1144 1793 1662 1106 8343 2029 5151 1107 1103 10250 4291 117 1133 1199 10301 125 117 1288 10992 1104 5671 1110 1286 1481 1451 1285 117 175 12831 1158 1107 1103 3208 1112 1122 24344 1111 1800 1106 2330 1122 1146 119 1135 1110 1510 1103 1234 1107 1103 2869 2556 11685 1116 1115 1132 4997 4634 119 1252 1107 1199 1877 1152 1132 2935 1171 119 1130 102 1405 1550 10992 1104 16259 26652 1132 1666 3828 1107 11086 119 102
03/15/2026 09:28:57 - INFO - utils_glue -   input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
03/15/2026 09:28:57 - INFO - utils_glue -   segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1
03/15/2026 09:28:57 - INFO - utils_glue -   label: not_entailment (id = 1)
03/15/2026 09:28:57 - INFO - utils_glue -   *** Example ***
03/15/2026 09:28:57 - INFO - utils_glue -   guid: dev-3
03/15/2026 09:28:57 - INFO - utils_glue -   tokens: [CLS] The Am ##ish community in Pennsylvania , which numbers about 55 , 000 , lives an a ##grarian lifestyle , s ##hun ##ning technological advances like electricity and automobiles . And many say their ins ##ular lifestyle gives them a sense that they are protected from the violence of American society . But as residents gathered near the school , some wearing traditional g ##ar ##b and arriving in horse - drawn bug ##gies , they said that sense of safety had been shattered . " If someone snaps and wants to do something stupid , there ' s no distance that ' s going to stop them , " said Jake [SEP] Pennsylvania has the biggest Am ##ish community in the U . S . [SEP]
03/15/2026 09:28:57 - INFO - utils_glue -   input_ids: 101 1109 7277 2944 1661 1107 2680 117 1134 2849 1164 3731 117 1288 117 2491 1126 170 25873 9897 117 188 17315 3381 12675 11823 1176 6495 1105 23338 119 1262 1242 1474 1147 22233 5552 9897 3114 1172 170 2305 1115 1152 1132 4921 1121 1103 4289 1104 1237 2808 119 1252 1112 3159 5260 1485 1103 1278 117 1199 3351 2361 176 1813 1830 1105 7190 1107 3241 118 3795 15430 19310 117 1152 1163 1115 2305 1104 3429 1125 1151 11670 119 107 1409 1800 22675 1105 3349 1106 1202 1380 4736 117 1175 112 188 1185 2462 1115 112 188 1280 1106 1831 1172 117 107 1163 4387 102 2680 1144 1103 4583 7277 2944 1661 1107 1103 158 119 156 119 102
03/15/2026 09:28:57 - INFO - utils_glue -   input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
03/15/2026 09:28:57 - INFO - utils_glue -   segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1
03/15/2026 09:28:57 - INFO - utils_glue -   label: not_entailment (id = 1)
03/15/2026 09:28:57 - INFO - utils_glue -   *** Example ***
03/15/2026 09:28:57 - INFO - utils_glue -   guid: dev-4
03/15/2026 09:28:57 - INFO - utils_glue -   tokens: [CLS] Security forces were on high alert after an election campaign in which more than 1 , 000 people , including seven election candidates , have been killed . [SEP] Security forces were on high alert after a campaign marred by violence . [SEP]
03/15/2026 09:28:57 - INFO - utils_glue -   input_ids: 101 4354 2088 1127 1113 1344 10427 1170 1126 1728 2322 1107 1134 1167 1190 122 117 1288 1234 117 1259 1978 1728 4765 117 1138 1151 1841 119 102 4354 2088 1127 1113 1344 10427 1170 170 2322 27767 1118 4289 119 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
03/15/2026 09:28:57 - INFO - utils_glue -   input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
03/15/2026 09:28:57 - INFO - utils_glue -   segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
03/15/2026 09:28:57 - INFO - utils_glue -   label: entailment (id = 0)
03/15/2026 09:28:58 - INFO - __main__ -   Saving features into cached file /users/jl0796/glue_data/RTE/cached_dev_bert-base-cased_128_rte
03/15/2026 09:28:58 - INFO - __main__ -   ***** Running evaluation  *****
03/15/2026 09:28:58 - INFO - __main__ -     Num examples = 277
03/15/2026 09:28:58 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:12<00:00,  2.88it/s]
03/15/2026 09:29:10 - INFO - __main__ -   ***** Eval results  *****███████████████████████████████████████████████████████████████████████████| 35/35 [00:12<00:00,  3.16it/s]
03/15/2026 09:29:10 - INFO - __main__ -     acc = 0.628158844765343
{'acc': np.float64(0.628158844765343)}
Iteration: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [05:18<00:00,  8.16s/it]
03/15/2026 09:34:29 - INFO - __main__ -   Loading features from cached file /users/jl0796/glue_data/RTE/cached_dev_bert-base-cased_128_rte████| 39/39 [05:18<00:00,  7.88s/it]
/users/jl0796/COS568-DistLM-SP26/run_glue.py:244: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  features = torch.load(cached_features_file)
03/15/2026 09:34:29 - INFO - __main__ -   ***** Running evaluation  *****
03/15/2026 09:34:29 - INFO - __main__ -     Num examples = 277
03/15/2026 09:34:29 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:12<00:00,  2.88it/s]
03/15/2026 09:34:41 - INFO - __main__ -   ***** Eval results  *****███████████████████████████████████████████████████████████████████████████| 35/35 [00:12<00:00,  3.13it/s]
03/15/2026 09:34:41 - INFO - __main__ -     acc = 0.6498194945848376
{'acc': np.float64(0.6498194945848376)}
Epoch:  67%|███████████████████████████████████████████████████████████████████████████████████████▎                                           | 2/3 [11:19<05:38, 338.
Iteration:  23%|█████████████████████████████▎                                                                                                 | 9/39 [01:13<04:02,  8.
Iteration:  26%|████████████████████████████████▎                                                                                             | 10/39 [01:21<03:55,  8.
Iteration:  28%|███████████████████████████████████▌                                                                                          | 11/39 [01:29<03:46,  8.
Iteration:  31%|██████████████████████████████████████▊                                                                                       | 12/39 [01:37<03:40,  8.
Iteration:  33%|██████████████████████████████████████████                                                                                    | 13/39 [01:45<03:30,  8.
Iteration:  36%|█████████████████████████████████████████████▏                                                                                | 14/39 [01:53<03:22,  8.
Iteration:  38%|████████████████████████████████████████████████▍                                                                             | 15/39 [02:02<03:14,  8.
Iteration:  41%|███████████████████████████████████████████████████▋                                                                          | 16/39 [02:10<03:07,  8.
Iteration:  44%|██████████████████████████████████████████████████████▉                                                                       | 17/39 [02:18<02:59,  8.
Iteration:  46%|██████████████████████████████████████████████████████████▏                                                                   | 18/39 [02:26<02:51,  8.
Iteration:  49%|█████████████████████████████████████████████████████████████▍                                                                | 19/39 [02:34<02:44,  8.
Iteration:  51%|████████████████████████████████████████████████████████████████▌                                                             | 20/39 [02:42<02:35,  8.
Iteration:  54%|███████████████████████████████████████████████████████████████████▊                                                          | 21/39 [02:51<02:29,  8.
Iteration:  56%|███████████████████████████████████████████████████████████████████████                                                       | 22/39 [02:59<02:19,  8.
Iteration:  59%|██████████████████████████████████████████████████████████████████████████▎                                                   | 23/39 [03:07<02:10,  8.
Iteration:  62%|█████████████████████████████████████████████████████████████████████████████▌                                                | 24/39 [03:15<02:02,  8.
Iteration:  64%|████████████████████████████████████████████████████████████████████████████████▊                                             | 25/39 [03:23<01:53,  8.
Iteration:  67%|████████████████████████████████████████████████████████████████████████████████████                                          | 26/39 [03:32<01:46,  8.
Iteration:  69%|███████████████████████████████████████████████████████████████████████████████████████▏                                      | 27/39 [03:40<01:37,  8.
Iteration:  72%|██████████████████████████████████████████████████████████████████████████████████████████▍                                   | 28/39 [03:48<01:29,  8.
Iteration:  74%|█████████████████████████████████████████████████████████████████████████████████████████████▋                                | 29/39 [03:56<01:21,  8.
Iteration:  77%|████████████████████████████████████████████████████████████████████████████████████████████████▉                             | 30/39 [04:04<01:13,  8.
Iteration:  79%|████████████████████████████████████████████████████████████████████████████████████████████████████▏                         | 31/39 [04:12<01:04,  8.
Iteration:  82%|███████████████████████████████████████████████████████████████████████████████████████████████████████▍                      | 32/39 [04:20<00:56,  8.
Iteration:  85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▌                   | 33/39 [04:28<00:48,  8.
Iteration:  87%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                | 34/39 [04:37<00:41,  8.
Iteration:  90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████             | 35/39 [04:45<00:32,  8.
Iteration:  92%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎         | 36/39 [04:53<00:24,  8.
Iteration:  95%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌      | 37/39 [05:01<00:16,  8.
Iteration:  97%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊   | 38/39 [05:09<00:08,  8.
Iteration: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [05:16<00:00,  8.12s/it]
03/15/2026 09:39:58 - INFO - __main__ -   Loading features from cached file /users/jl0796/glue_data/RTE/cached_dev_bert-base-cased_128_rte
03/15/2026 09:39:58 - INFO - __main__ -   ***** Running evaluation  *****
03/15/2026 09:39:58 - INFO - __main__ -     Num examples = 277
03/15/2026 09:39:58 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:12<00:00,  2.91it/s]
03/15/2026 09:40:10 - INFO - __main__ -   ***** Eval results  *****████████████████████████████████████████████████████████████████████| 35/35 [00:12<00:00,  3.18it/s]
03/15/2026 09:40:10 - INFO - __main__ -     acc = 0.6209386281588448
{'acc': np.float64(0.6209386281588448)}
Epoch: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [16:48<00:00, 333.Epoch: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [16:48<00:00, 336.18s/it]
03/15/2026 09:40:10 - INFO - __main__ -    global_step = 117, average loss = 0.6365272191345183
03/15/2026 09:40:10 - INFO - __main__ -   Loading features from cached file /users/jl0796/glue_data/RTE/cached_dev_bert-base-cased_128_rte
03/15/2026 09:40:10 - INFO - __main__ -   ***** Running evaluation  *****
03/15/2026 09:40:10 - INFO - __main__ -     Num examples = 277
03/15/2026 09:40:10 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:12<00:00,  2.89it/s]
03/15/2026 09:40:22 - INFO - __main__ -   ***** Eval results  *****
03/15/2026 09:40:22 - INFO - __main__ -     acc = 0.6209386281588448



Rerun on each new node:
 sudo apt-get update
 sudo apt-get install htop dstat python3-pip
 echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
 source ~/.bashrc
 
 pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
 pip install numpy scipy scikit-learn tqdm pytorch_transformers apex
 cd /users/jl0796/
 git clone https://github.com/liujonathan24/COS568-DistLM-SP26.git
 cd COS568-DistLM-SP26
 python3 download_glue_data.py
