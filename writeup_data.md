task 3:
```
cd /users/jl0796/COS568-DistLM-SP26/
python download_glue_data.py
cd /users/jl0796/COS568-DistLM-SP26/task2b
export GLUE_DIR=$HOME/COS568-DistLM-SP26/glue_data
export TASK_NAME=RTE
export IP_ADDRESS=128.110.218.95
export PORT=13579
python3 run_glue.py   --model_type bert   --model_name_or_path bert-base-cased   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128   --per_device_train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 3   --output_dir /tmp/$TASK_NAME/   --overwrite_output_dir \
--master_ip $IP_ADDRESS --master_port $PORT --world_size 4 --local_rank [0-3]
```

rank0:
  Rank 0 Step 1: 0.8928709030151367 loss, 2.8005s                              | 1/39 [00:05<03:17,  5.21s/it]
                                                                                                          Rank 0 Step 2: 0.6499286890029907 loss, 2.7297s                              | 2/39 [00:08<02:23,  3.89s/it]
                                                                                                          Rank 0 Step 3: 0.6082603335380554 loss, 2.7265s                              | 3/39 [00:11<02:03,  3.43s/it]
                                                                                                          Rank 0 Step 4: 0.7481549382209778 loss, 2.7764s                              | 4/39 [00:15<02:07,  3.64s/it]
                                                                                                          Rank 0 Step 5: 0.7163246870040894 loss, 2.7312s                              | 5/39 [00:17<01:54,  3.38s/it]
                                                                                                          Rank 0 Step 6: 0.712179958820343 loss, 2.7181s                               | 6/39 [00:20<01:45,  3.21s/it]
                                                                                                          Rank 0 Step 7: 0.7204083204269409 loss, 2.6951s                              | 7/39 [00:23<01:39,  3.10s/it]
                                                                                                          Rank 0 Step 8: 0.6837247610092163 loss, 2.6330s                              | 8/39 [00:26<01:33,  3.01s/it]
                                                                                                          Rank 0 Step 9: 0.7025787830352783 loss, 2.6398s                              | 9/39 [00:29<01:28,  2.94s/it]
                                                                                                          Rank 0 Step 10: 0.6743227243423462 loss, 2.6115s                            | 10/39 [00:32<01:23,  2.89s/it]
                                                                                                          Rank 0 Step 11: 0.6346275806427002 loss, 2.6817s                            | 11/39 [00:34<01:19,  2.84s/it]
                                                                                                          Rank 0 Step 12: 0.7356266379356384 loss, 2.6840s                            | 12/39 [00:37<01:16,  2.84s/it]
                                                                                                          Rank 0 Step 13: 0.7202005982398987 loss, 2.6024s                            | 13/39 [00:40<01:13,  2.83s/it]
                                                                                                          Rank 0 Step 14: 0.6969020366668701 loss, 2.6146s                            | 14/39 [00:43<01:10,  2.80s/it]
                                                                                                          Rank 0 Step 15: 0.6632838249206543 loss, 2.6193s                            | 15/39 [00:45<01:06,  2.79s/it]
                                                                                                          Rank 0 Step 16: 0.7012203335762024 loss, 2.6381s                            | 16/39 [00:48<01:03,  2.78s/it]
                                                                                                          Rank 0 Step 17: 0.6528066992759705 loss, 2.6174s                            | 17/39 [00:51<01:01,  2.78s/it]
                                                                                                          Rank 0 Step 18: 0.6856709122657776 loss, 2.6717s                            | 18/39 [00:54<00:58,  2.77s/it]
                                                                                                          Rank 0 Step 19: 0.6888301968574524 loss, 2.6606s                            | 19/39 [00:57<00:55,  2.78s/it]
                                                                                                          Rank 0 Step 20: 0.6908309459686279 loss, 2.7099s                            | 20/39 [00:59<00:52,  2.79s/it]
                                                                                                          Rank 0 Step 21: 0.6993993520736694 loss, 2.6509s▊                           | 21/39 [01:02<00:50,  2.81s/it]
                                                                                                          Rank 0 Step 22: 0.6756142377853394 loss, 2.6558s██▎                         | 22/39 [01:05<00:47,  2.80s/it]
                                                                                                          Rank 0 Step 23: 0.668753981590271 loss, 2.6933s████▊                        | 23/39 [01:08<00:44,  2.80s/it]
                                                                                                          Rank 0 Step 24: 0.6765525937080383 loss, 2.6153s█████▎                      | 24/39 [01:11<00:42,  2.81s/it]
                                                                                                          Rank 0 Step 25: 0.6721910834312439 loss, 2.6235s██████▊                     | 25/39 [01:13<00:39,  2.79s/it]
                                                                                                          Rank 0 Step 26: 0.6468062996864319 loss, 2.6397s████████▎                   | 26/39 [01:16<00:36,  2.78s/it]
                                                                                                          Rank 0 Step 27: 0.6363000273704529 loss, 2.7015s█████████▊                  | 27/39 [01:19<00:33,  2.78s/it]
                                                                                                          Rank 0 Step 28: 0.6307457685470581 loss, 2.6236s███████████▎                | 28/39 [01:22<00:30,  2.80s/it]
                                                                                                          Rank 0 Step 29: 0.6405341029167175 loss, 2.6637s████████████▊               | 29/39 [01:24<00:27,  2.79s/it]
                                                                                                          Rank 0 Step 30: 0.7167795896530151 loss, 2.6755s██████████████▍             | 30/39 [01:27<00:25,  2.79s/it]
                                                                                                          Rank 0 Step 31: 0.6874585747718811 loss, 2.6699s███████████████▉            | 31/39 [01:30<00:22,  2.80s/it]
                                                                                                          Rank 0 Step 32: 0.6785904169082642 loss, 2.6377s█████████████████▍          | 32/39 [01:33<00:19,  2.80s/it]
                                                                                                          Rank 0 Step 33: 0.7177387475967407 loss, 2.6128s██████████████████▉         | 33/39 [01:36<00:16,  2.79s/it]
                                                                                                          Rank 0 Step 34: 0.7194638252258301 loss, 2.6494s████████████████████▍       | 34/39 [01:38<00:13,  2.78s/it]
                                                                                                          Rank 0 Step 35: 0.6533253192901611 loss, 2.6499s█████████████████████▉      | 35/39 [01:41<00:11,  2.78s/it]
                                                                                                          Rank 0 Step 36: 0.6592704057693481 loss, 2.6575s███████████████████████▍    | 36/39 [01:44<00:08,  2.78s/it]
                                                                                                          Rank 0 Step 37: 0.7123329043388367 loss, 2.6257s████████████████████████▉   | 37/39 [01:47<00:05,  2.79s/it]
                                                                                                          Rank 0 Step 38: 0.6418021321296692 loss, 2.4812s██████████████████████████▍ | 38/39 [01:50<00:02,  2.78s/it]
Iteration: 100%|███████████████████████████████████████████████████████████| 39/39 [01:52<00:00,  2.89s/it]
03/19/2026 15:23:54 - INFO - __main__ -   Loading features from cached file /users/jl0796/COS568-DistLM-SP26/glue_data/RTE/cached_dev_bert-base-cased_128_rte
03/19/2026 15:23:54 - INFO - __main__ -   ***** Running evaluation  *****
03/19/2026 15:23:54 - INFO - __main__ -     Num examples = 277
03/19/2026 15:23:54 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.98it/s]
03/19/2026 15:23:57 - INFO - __main__ -   ***** Eval results  *****██████████| 9/9 [00:03<00:00,  3.14it/s]
03/19/2026 15:23:57 - INFO - __main__ -     acc = 0.5285714285714286
{'acc': np.float64(0.5285714285714286)}
Epoch: 100%|████████████████████████████████████████████████████████████████| 1/1 [01:55<00:00, 115.72s/it]
03/19/2026 15:23:58 - INFO - __main__ -    global_step = 39, average loss = 0.6936845259788709
03/19/2026 15:23:58 - INFO - __main__ -   Loading features from cached file /users/jl0796/COS568-DistLM-SP26/glue_data/RTE/cached_dev_bert-base-cased_128_rte
03/19/2026 15:23:58 - INFO - __main__ -   ***** Running evaluation  *****
03/19/2026 15:23:58 - INFO - __main__ -     Num examples = 277
03/19/2026 15:23:58 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.99it/s]
03/19/2026 15:24:01 - INFO - __main__ -   ***** Eval results  *****
03/19/2026 15:24:01 - INFO - __main__ -     acc = 0.5285714285714286

rank1:
Rank 1 Step 1: 0.7918104529380798 loss, 2.8108s
Rank 1 Step 2: 0.7261569499969482 loss, 2.7294s
Rank 1 Step 3: 0.6006006598472595 loss, 2.7279s
Rank 1 Step 4: 0.671589732170105 loss, 2.7846s
Rank 1 Step 5: 0.649652361869812 loss, 2.7246s
Rank 1 Step 6: 0.6816576719284058 loss, 2.7368s
Rank 1 Step 7: 0.7670220732688904 loss, 2.6887s
Rank 1 Step 8: 0.7426831126213074 loss, 2.6311s
Rank 1 Step 9: 0.6736165285110474 loss, 2.6414s
Rank 1 Step 10: 0.6864630579948425 loss, 2.6100s
Rank 1 Step 11: 0.7301202416419983 loss, 2.6784s
Rank 1 Step 12: 0.6975117325782776 loss, 2.6877s
Rank 1 Step 13: 0.6921250820159912 loss, 2.6026s
Rank 1 Step 14: 0.7272283434867859 loss, 2.6145s
Rank 1 Step 15: 0.6534180045127869 loss, 2.6180s
Rank 1 Step 16: 0.6618313789367676 loss, 2.6393s
Rank 1 Step 17: 0.7106463313102722 loss, 2.6155s
Rank 1 Step 18: 0.6821306943893433 loss, 2.6725s
Rank 1 Step 19: 0.6719106435775757 loss, 2.6597s
Rank 1 Step 20: 0.6581270694732666 loss, 2.7109s
Rank 1 Step 21: 0.6778276562690735 loss, 2.6483s
Rank 1 Step 22: 0.634501576423645 loss, 2.6548s
Rank 1 Step 23: 0.6647440791130066 loss, 2.6918s
Rank 1 Step 24: 0.6870583891868591 loss, 2.6163s
Rank 1 Step 25: 0.6760925054550171 loss, 2.6239s
Rank 1 Step 26: 0.7045295238494873 loss, 2.6393s
Rank 1 Step 27: 0.7360819578170776 loss, 2.6950s
Rank 1 Step 28: 0.7226952910423279 loss, 2.6384s
Rank 1 Step 29: 0.658108115196228 loss, 2.6613s
Rank 1 Step 30: 0.7169983983039856 loss, 2.6746s
Rank 1 Step 31: 0.6600024700164795 loss, 2.6675s
Rank 1 Step 32: 0.6641342043876648 loss, 2.6410s
Rank 1 Step 33: 0.7266454100608826 loss, 2.6115s
Rank 1 Step 34: 0.7093887329101562 loss, 2.6510s
Rank 1 Step 35: 0.6652270555496216 loss, 2.6491s
Rank 1 Step 36: 0.6681618690490723 loss, 2.6532s
Rank 1 Step 37: 0.689782977104187 loss, 2.6271s
Rank 1 Step 38: 0.7069808840751648 loss, 2.4786s
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.96it/s]
{'acc': np.float64(0.7285714285714285)}


rank2:
Rank 2 Step 1: 0.6797139048576355 loss, 2.7999s
Rank 2 Step 2: 0.7076496481895447 loss, 2.7309s
Rank 2 Step 3: 0.6400177478790283 loss, 2.7274s
Rank 2 Step 4: 0.7028261423110962 loss, 2.8055s
Rank 2 Step 5: 0.7736439108848572 loss, 2.7300s
Rank 2 Step 6: 0.6513296365737915 loss, 2.7317s
Rank 2 Step 7: 0.751022219657898 loss, 2.6890s
Rank 2 Step 8: 0.6548188328742981 loss, 2.6317s
Rank 2 Step 9: 0.707629382610321 loss, 2.6384s
Rank 2 Step 10: 0.6588904857635498 loss, 2.6097s
Rank 2 Step 11: 0.6145505309104919 loss, 2.6798s
Rank 2 Step 12: 0.6967117786407471 loss, 2.6870s
Rank 2 Step 13: 0.6983460783958435 loss, 2.6041s
Rank 2 Step 14: 0.64417964220047 loss, 2.6153s
Rank 2 Step 15: 0.6789699196815491 loss, 2.6163s
Rank 2 Step 16: 0.6519253849983215 loss, 2.6398s
Rank 2 Step 17: 0.7012290954589844 loss, 2.6178s
Rank 2 Step 18: 0.691001296043396 loss, 2.6712s
Rank 2 Step 19: 0.6754285097122192 loss, 2.6602s
Rank 2 Step 20: 0.6873687505722046 loss, 2.7127s
Rank 2 Step 21: 0.7326023578643799 loss, 2.6472s
Rank 2 Step 22: 0.6204068660736084 loss, 2.6555s
Rank 2 Step 23: 0.6833940148353577 loss, 2.6930s
Rank 2 Step 24: 0.6525702476501465 loss, 2.6163s
Rank 2 Step 25: 0.6798869967460632 loss, 2.6244s
Rank 2 Step 26: 0.7147308588027954 loss, 2.6373s
Rank 2 Step 27: 0.723520815372467 loss, 2.6996s
Rank 2 Step 28: 0.6681506633758545 loss, 2.6357s
Rank 2 Step 29: 0.697526752948761 loss, 2.6628s
Rank 2 Step 30: 0.6580085754394531 loss, 2.6728s
Rank 2 Step 31: 0.6505770087242126 loss, 2.6719s
Rank 2 Step 32: 0.6886754631996155 loss, 2.6380s
Rank 2 Step 33: 0.6778713464736938 loss, 2.6156s
Rank 2 Step 34: 0.7074357271194458 loss, 2.6474s
Rank 2 Step 35: 0.6593215465545654 loss, 2.6492s
Rank 2 Step 36: 0.681056022644043 loss, 2.6604s
Rank 2 Step 37: 0.695023238658905 loss, 2.6262s
Rank 2 Step 38: 0.7639906406402588 loss, 2.4686s
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.96it/s]
{'acc': np.float64(0.5857142857142857)}

rank3:
Rank 3 Step 1: 0.6271342039108276 loss, 2.8020s
Rank 3 Step 2: 0.733111560344696 loss, 2.7326s
Rank 3 Step 3: 0.7530367970466614 loss, 2.7307s
Rank 3 Step 4: 0.7701762318611145 loss, 2.7694s
Rank 3 Step 5: 0.7852979302406311 loss, 2.7309s
Rank 3 Step 6: 0.6773862838745117 loss, 2.7324s
Rank 3 Step 7: 0.7240059971809387 loss, 2.6931s
Rank 3 Step 8: 0.6956485509872437 loss, 2.6320s
Rank 3 Step 9: 0.7058278322219849 loss, 2.6383s
Rank 3 Step 10: 0.7347922921180725 loss, 2.6108s
Rank 3 Step 11: 0.7404108047485352 loss, 2.6812s
Rank 3 Step 12: 0.6911787390708923 loss, 2.6863s
Rank 3 Step 13: 0.6838893890380859 loss, 2.6052s
Rank 3 Step 14: 0.6942996382713318 loss, 2.6132s
Rank 3 Step 15: 0.723535418510437 loss, 2.6192s
Rank 3 Step 16: 0.6867842674255371 loss, 2.6413s
Rank 3 Step 17: 0.7014889717102051 loss, 2.6087s
Rank 3 Step 18: 0.6868188381195068 loss, 2.6681s
Rank 3 Step 19: 0.6634982824325562 loss, 2.6614s
Rank 3 Step 20: 0.6566498279571533 loss, 2.7091s
Rank 3 Step 21: 0.6318873167037964 loss, 2.6517s
Rank 3 Step 22: 0.7325892448425293 loss, 2.6577s
Rank 3 Step 23: 0.66378253698349 loss, 2.6924s
Rank 3 Step 24: 0.6736941933631897 loss, 2.6179s
Rank 3 Step 25: 0.7446627616882324 loss, 2.6246s
Rank 3 Step 26: 0.7009887099266052 loss, 2.6379s
Rank 3 Step 27: 0.6627845764160156 loss, 2.6969s
Rank 3 Step 28: 0.7231649160385132 loss, 2.6361s
Rank 3 Step 29: 0.6134838461875916 loss, 2.6632s
Rank 3 Step 30: 0.6876940727233887 loss, 2.6726s
Rank 3 Step 31: 0.6324418783187866 loss, 2.6707s
Rank 3 Step 32: 0.6902807950973511 loss, 2.6419s
Rank 3 Step 33: 0.6070387363433838 loss, 2.6156s
Rank 3 Step 34: 0.6718780398368835 loss, 2.6485s
Rank 3 Step 35: 0.6596216559410095 loss, 2.6484s
Rank 3 Step 36: 0.6989961862564087 loss, 2.6603s
Rank 3 Step 37: 0.6583592891693115 loss, 2.6260s
Rank 3 Step 38: 0.6860821843147278 loss, 2.4798s
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.96it/s]
{'acc': np.float64(0.5428571428571428)}

task 2b:
- python run_glue.py [other input args] --master_ip $ip_address$ --master_port $port$ --world_size 4 --local_rank $rank$

```
cd /users/jl0796/COS568-DistLM-SP26/
python download_glue_data.py
cd /users/jl0796/COS568-DistLM-SP26/task2b
export GLUE_DIR=$HOME/COS568-DistLM-SP26/glue_data
export TASK_NAME=RTE
export IP_ADDRESS=128.110.218.95
export PORT=13579
python3 run_glue.py   --model_type bert   --model_name_or_path bert-base-cased   --task_name $TASK_NAME   --do_train   --do_eval   --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128   --per_device_train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 3   --output_dir /tmp/$TASK_NAME/   --overwrite_output_dir \
--master_ip $IP_ADDRESS --master_port $PORT --world_size 4 --local_rank [0-3]
```
rank0:
 Rank 0 Step 1: 0.9275242686271667 loss, 3.2632s                              | 1/39 [00:05<03:21,  5.29s/it]
                                                                                                          Rank 0 Step 2: 0.6337472796440125 loss, 3.2469s                              | 2/39 [00:08<02:35,  4.20s/it]
                                                                                                          Rank 0 Step 3: 0.5745732188224792 loss, 3.2681s                              | 3/39 [00:12<02:18,  3.85s/it]
                                                                                                          Rank 0 Step 4: 0.7925540208816528 loss, 3.2645s                              | 4/39 [00:16<02:25,  4.15s/it]
                                                                                                          Rank 0 Step 5: 0.7361527681350708 loss, 3.1043s                              | 5/39 [00:20<02:12,  3.90s/it]
                                                                                                          Rank 0 Step 6: 0.7480979561805725 loss, 3.1026s                              | 6/39 [00:23<02:01,  3.68s/it]
                                                                                                          Rank 0 Step 7: 0.7045212388038635 loss, 3.0957s                              | 7/39 [00:26<01:53,  3.54s/it]
                                                                                                          Rank 0 Step 8: 0.7056183218955994 loss, 3.0700s                              | 8/39 [00:30<01:47,  3.45s/it]
                                                                                                          Rank 0 Step 9: 0.6485725045204163 loss, 3.0947s                              | 9/39 [00:33<01:41,  3.38s/it]
                                                                                                          Rank 0 Step 10: 0.7597270011901855 loss, 3.1452s                            | 10/39 [00:36<01:36,  3.34s/it]
                                                                                                          Rank 0 Step 11: 0.7363904714584351 loss, 3.1353s                            | 11/39 [00:39<01:33,  3.33s/it]
                                                                                                          Rank 0 Step 12: 0.7677759528160095 loss, 3.0892s                            | 12/39 [00:43<01:29,  3.32s/it]
                                                                                                          Rank 0 Step 13: 0.7352871298789978 loss, 3.0996s                            | 13/39 [00:46<01:25,  3.30s/it]
                                                                                                          Rank 0 Step 14: 0.7306473255157471 loss, 3.1159s                            | 14/39 [00:49<01:22,  3.28s/it]
                                                                                                          Rank 0 Step 15: 0.6343802809715271 loss, 3.1043s                            | 15/39 [00:52<01:18,  3.28s/it]
                                                                                                          Rank 0 Step 16: 0.7133766412734985 loss, 3.0865s                            | 16/39 [00:56<01:15,  3.27s/it]
                                                                                                          Rank 0 Step 17: 0.6251988410949707 loss, 3.1241s                            | 17/39 [00:59<01:11,  3.26s/it]
                                                                                                          Rank 0 Step 18: 0.692106306552887 loss, 3.0826s                             | 18/39 [01:02<01:08,  3.27s/it]
                                                                                                          Rank 0 Step 19: 0.6669694781303406 loss, 3.0773s                            | 19/39 [01:05<01:05,  3.27s/it]
                                                                                                          Rank 0 Step 20: 0.7072840332984924 loss, 3.1142s                            | 20/39 [01:09<01:01,  3.26s/it]
                                                                                                          Rank 0 Step 21: 0.6922056674957275 loss, 3.1377s▊                           | 21/39 [01:12<00:58,  3.26s/it]
                                                                                                          Rank 0 Step 22: 0.6656978130340576 loss, 3.0765s██▎                         | 22/39 [01:15<00:55,  3.27s/it]
                                                                                                          Rank 0 Step 23: 0.6616635322570801 loss, 3.1155s███▊                        | 23/39 [01:18<00:52,  3.26s/it]
                                                                                                          Rank 0 Step 24: 0.6690232157707214 loss, 3.1337s█████▎                      | 24/39 [01:22<00:48,  3.26s/it]
                                                                                                          Rank 0 Step 25: 0.6990297436714172 loss, 3.1108s██████▊                     | 25/39 [01:25<00:45,  3.27s/it]
                                                                                                          Rank 0 Step 26: 0.6498892307281494 loss, 3.1107s████████▎                   | 26/39 [01:28<00:42,  3.27s/it]
                                                                                                          Rank 0 Step 27: 0.6175944805145264 loss, 3.1232s█████████▊                  | 27/39 [01:32<00:39,  3.27s/it]
                                                                                                          Rank 0 Step 28: 0.6360681653022766 loss, 3.0837s███████████▎                | 28/39 [01:35<00:36,  3.27s/it]
                                                                                                          Rank 0 Step 29: 0.6467706561088562 loss, 3.0728s████████████▊               | 29/39 [01:38<00:32,  3.26s/it]
                                                                                                          Rank 0 Step 30: 0.713947057723999 loss, 3.0543s███████████████▍             | 30/39 [01:41<00:29,  3.25s/it]
                                                                                                          Rank 0 Step 31: 0.666487455368042 loss, 3.0945s████████████████▉            | 31/39 [01:44<00:25,  3.23s/it]
                                                                                                          Rank 0 Step 32: 0.6726608276367188 loss, 3.0793s█████████████████▍          | 32/39 [01:48<00:22,  3.24s/it]
                                                                                                          Rank 0 Step 33: 0.6935362815856934 loss, 3.1406s██████████████████▉         | 33/39 [01:51<00:19,  3.24s/it]
                                                                                                          Rank 0 Step 34: 0.6906305551528931 loss, 3.0874s████████████████████▍       | 34/39 [01:54<00:16,  3.26s/it]
                                                                                                          Rank 0 Step 35: 0.6484131813049316 loss, 3.0781s█████████████████████▉      | 35/39 [01:58<00:13,  3.25s/it]
                                                                                                          Rank 0 Step 36: 0.656643271446228 loss, 3.0619s████████████████████████▍    | 36/39 [02:01<00:09,  3.25s/it]
                                                                                                          Rank 0 Step 37: 0.7162936925888062 loss, 3.0882s████████████████████████▉   | 37/39 [02:04<00:06,  3.24s/it]
                                                                                                          Rank 0 Step 38: 0.6618431210517883 loss, 2.9405s██████████████████████████▍ | 38/39 [02:07<00:03,  3.24s/it]
Iteration: 100%|███████████████████████████████████████████████████████████| 39/39 [02:10<00:00,  3.35s/it]

03/19/2026 15:00:07 - INFO - __main__ -   ***** Running evaluation  *****
03/19/2026 15:00:07 - INFO - __main__ -     Num examples = 277
03/19/2026 15:00:07 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.96it/s]
03/19/2026 15:00:10 - INFO - __main__ -   ***** Eval results  *****██████████| 9/9 [00:03<00:00,  3.13it/s]
03/19/2026 15:00:10 - INFO - __main__ -     acc = 0.5714285714285714
{'acc': np.float64(0.5714285714285714)}
Epoch: 100%|████████████████████████████████████████████████████████████████| 1/1 [02:13<00:00, 133.85s/it]
03/19/2026 15:00:10 - INFO - __main__ -    global_step = 39, average loss = 0.6984663147192735
03/19/2026 15:00:10 - INFO - __main__ -   Loading features from cached file /users/jl0796/COS568-DistLM-SP26/glue_data/RTE/cached_dev_bert-base-cased_128_rte
03/19/2026 15:00:10 - INFO - __main__ -   ***** Running evaluation  *****
03/19/2026 15:00:10 - INFO - __main__ -     Num examples = 277
03/19/2026 15:00:10 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.97it/s]
03/19/2026 15:00:14 - INFO - __main__ -   ***** Eval results  *****
03/19/2026 15:00:14 - INFO - __main__ -     acc = 0.5714285714285714


rank1:;
Rank 1 Step 1: 0.8088881969451904 loss, 3.2595s
Rank 1 Step 2: 0.7751754522323608 loss, 3.2352s
Rank 1 Step 3: 0.6154294610023499 loss, 3.2829s
Rank 1 Step 4: 0.732525646686554 loss, 3.2847s
Rank 1 Step 5: 0.6826475858688354 loss, 3.1246s
Rank 1 Step 6: 0.7972484827041626 loss, 3.1090s
Rank 1 Step 7: 0.8497341871261597 loss, 3.0959s
Rank 1 Step 8: 0.7141337394714355 loss, 3.0915s
Rank 1 Step 9: 0.7543343901634216 loss, 3.0995s
Rank 1 Step 10: 0.8139281868934631 loss, 3.1196s
Rank 1 Step 11: 0.7448554635047913 loss, 3.1527s
Rank 1 Step 12: 0.7255043387413025 loss, 3.0913s
Rank 1 Step 13: 0.6903994083404541 loss, 3.1213s
Rank 1 Step 14: 0.7242132425308228 loss, 3.1159s
Rank 1 Step 15: 0.722054123878479 loss, 3.1046s
Rank 1 Step 16: 0.6147738695144653 loss, 3.0946s
Rank 1 Step 17: 0.7293050289154053 loss, 3.1234s
Rank 1 Step 18: 0.692111611366272 loss, 3.1047s
Rank 1 Step 19: 0.6828621625900269 loss, 3.1096s
Rank 1 Step 20: 0.647454559803009 loss, 3.0972s
Rank 1 Step 21: 0.6743173599243164 loss, 3.1385s
Rank 1 Step 22: 0.6162533760070801 loss, 3.0893s
Rank 1 Step 23: 0.6681492924690247 loss, 3.1061s
Rank 1 Step 24: 0.6844494342803955 loss, 3.1192s
Rank 1 Step 25: 0.6790174245834351 loss, 3.1025s
Rank 1 Step 26: 0.7332028150558472 loss, 3.1215s
Rank 1 Step 27: 0.7437059879302979 loss, 3.1366s
Rank 1 Step 28: 0.7416520714759827 loss, 3.0820s
Rank 1 Step 29: 0.6495732069015503 loss, 3.0677s
Rank 1 Step 30: 0.7171584367752075 loss, 3.0517s
Rank 1 Step 31: 0.6885585188865662 loss, 3.0893s
Rank 1 Step 32: 0.6243107914924622 loss, 3.0929s
Rank 1 Step 33: 0.7293746471405029 loss, 3.1501s
Rank 1 Step 34: 0.7114556431770325 loss, 3.0888s
Rank 1 Step 35: 0.6508676409721375 loss, 3.0791s
Rank 1 Step 36: 0.679631769657135 loss, 3.0714s
Rank 1 Step 37: 0.6973370909690857 loss, 3.0880s
Rank 1 Step 38: 0.7089685201644897 loss, 2.9603s
{'acc': np.float64(0.6714285714285714)}

rank2:
Rank 2 Step 1: 0.6893115043640137 loss, 3.2593s
Rank 2 Step 2: 0.6745951771736145 loss, 3.2490s
Rank 2 Step 3: 0.5807715654373169 loss, 3.2838s
Rank 2 Step 4: 0.6408743262290955 loss, 3.2945s
Rank 2 Step 5: 0.8026094436645508 loss, 3.1112s
Rank 2 Step 6: 0.6470295190811157 loss, 3.1077s
Rank 2 Step 7: 0.7225726246833801 loss, 3.0930s
Rank 2 Step 8: 0.765575110912323 loss, 3.0894s
Rank 2 Step 9: 0.6502664089202881 loss, 3.0959s
Rank 2 Step 10: 0.6749289035797119 loss, 3.1308s
Rank 2 Step 11: 0.6565289497375488 loss, 3.1486s
Rank 2 Step 12: 0.6667109727859497 loss, 3.0899s
Rank 2 Step 13: 0.7003048658370972 loss, 3.1207s
Rank 2 Step 14: 0.6759980916976929 loss, 3.1167s
Rank 2 Step 15: 0.7211638689041138 loss, 3.1013s
Rank 2 Step 16: 0.6674908995628357 loss, 3.0990s
Rank 2 Step 17: 0.6738566160202026 loss, 3.1224s
Rank 2 Step 18: 0.6872974038124084 loss, 3.1044s
Rank 2 Step 19: 0.6855839490890503 loss, 3.1097s
Rank 2 Step 20: 0.6864213347434998 loss, 3.1170s
Rank 2 Step 21: 0.7432588338851929 loss, 3.1409s
Rank 2 Step 22: 0.6011861562728882 loss, 3.0813s
Rank 2 Step 23: 0.679348886013031 loss, 3.1179s
Rank 2 Step 24: 0.6041138172149658 loss, 3.1269s
Rank 2 Step 25: 0.7135167717933655 loss, 3.1142s
Rank 2 Step 26: 0.7276502847671509 loss, 3.1180s
Rank 2 Step 27: 0.7308622002601624 loss, 3.1379s
Rank 2 Step 28: 0.6620176434516907 loss, 3.0715s
Rank 2 Step 29: 0.6779234409332275 loss, 3.0698s
Rank 2 Step 30: 0.6656278371810913 loss, 3.0450s
Rank 2 Step 31: 0.6437429785728455 loss, 3.0934s
Rank 2 Step 32: 0.6978836059570312 loss, 3.0839s
Rank 2 Step 33: 0.6921667456626892 loss, 3.1496s
Rank 2 Step 34: 0.7039715051651001 loss, 3.0862s
Rank 2 Step 35: 0.6654229760169983 loss, 3.0678s
Rank 2 Step 36: 0.6784185171127319 loss, 3.0657s
Rank 2 Step 37: 0.7009241580963135 loss, 3.0893s
Rank 2 Step 38: 0.7804263830184937 loss, 2.9540s
{'acc': np.float64(0.5714285714285714)}

rank3:
Rank 3 Step 1: 0.6378768682479858 loss, 3.2771s
Rank 3 Step 2: 0.7848365306854248 loss, 3.2519s
Rank 3 Step 3: 0.8473948240280151 loss, 3.2859s
Rank 3 Step 4: 0.762873649597168 loss, 3.2876s
Rank 3 Step 5: 0.7861863970756531 loss, 3.1378s
Rank 3 Step 6: 0.6439180970191956 loss, 3.1017s
Rank 3 Step 7: 0.7529059052467346 loss, 3.0940s
Rank 3 Step 8: 0.6252505779266357 loss, 3.1016s
Rank 3 Step 9: 0.6525262594223022 loss, 3.1101s
Rank 3 Step 10: 0.6911728978157043 loss, 3.1617s
Rank 3 Step 11: 0.7579160332679749 loss, 3.1646s
Rank 3 Step 12: 0.7595994472503662 loss, 3.1052s
Rank 3 Step 13: 0.7044042348861694 loss, 3.1318s
Rank 3 Step 14: 0.7202672362327576 loss, 3.1274s
Rank 3 Step 15: 0.6941282153129578 loss, 3.1171s
Rank 3 Step 16: 0.6632963418960571 loss, 3.1098s
Rank 3 Step 17: 0.6638643145561218 loss, 3.1367s
Rank 3 Step 18: 0.7128469944000244 loss, 3.1158s
Rank 3 Step 19: 0.6600682735443115 loss, 3.1189s
Rank 3 Step 20: 0.6472813487052917 loss, 3.1277s
Rank 3 Step 21: 0.6282678842544556 loss, 3.1372s
Rank 3 Step 22: 0.7439930438995361 loss, 3.1028s
Rank 3 Step 23: 0.6477342844009399 loss, 3.1324s
Rank 3 Step 24: 0.6826679706573486 loss, 3.1490s
Rank 3 Step 25: 0.7687020301818848 loss, 3.1276s
Rank 3 Step 26: 0.7249168157577515 loss, 3.1347s
Rank 3 Step 27: 0.6383755207061768 loss, 3.1533s
Rank 3 Step 28: 0.7754286527633667 loss, 3.0965s
Rank 3 Step 29: 0.5770162343978882 loss, 3.0852s
Rank 3 Step 30: 0.6482086181640625 loss, 3.0641s
Rank 3 Step 31: 0.6300311088562012 loss, 3.1051s
Rank 3 Step 32: 0.6769804954528809 loss, 3.1051s
Rank 3 Step 33: 0.5858184099197388 loss, 3.1625s
Rank 3 Step 34: 0.677280843257904 loss, 3.1054s
Rank 3 Step 35: 0.6427037715911865 loss, 3.0930s
Rank 3 Step 36: 0.6742522716522217 loss, 3.0857s
Rank 3 Step 37: 0.6562796831130981 loss, 3.1006s
Rank 3 Step 38: 0.6848655939102173 loss, 2.9703s
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.99it/s]
{'acc': np.float64(0.5571428571428572)}





task 2a:

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

ank 0 Step 1: 0.927524209022522 loss, 5.8778s                               | 1/39 [00:07<04:53,  7.72s/it]
                                                                                                          Rank 0 Step 2: 0.6337472796440125 loss, 6.0086s                              | 2/39 [00:13<04:09,  6.75s/it]
                                                                                                          Rank 0 Step 3: 0.5745731592178345 loss, 6.0268s                              | 3/39 [00:20<03:54,  6.50s/it]
                                                                                                          Rank 0 Step 4: 0.7925539612770081 loss, 6.0142s                              | 4/39 [00:28<04:09,  7.14s/it]
                                                                                                          Rank 0 Step 5: 0.7361527681350708 loss, 5.4703s                              | 5/39 [00:34<03:51,  6.80s/it]
                                                                                                          Rank 0 Step 6: 0.7480980157852173 loss, 5.5012s                              | 6/39 [00:39<03:31,  6.41s/it]
                                                                                                          Rank 0 Step 7: 0.7045212984085083 loss, 5.5018s                              | 7/39 [00:45<03:17,  6.16s/it]
                                                                                                          Rank 0 Step 8: 0.7056183218955994 loss, 5.5315s                              | 8/39 [00:51<03:06,  6.01s/it]
                                                                                                          Rank 0 Step 9: 0.648572564125061 loss, 5.5191s                               | 9/39 [00:57<02:57,  5.91s/it]
                                                                                                          Rank 0 Step 10: 0.7597270011901855 loss, 5.5368s                            | 10/39 [01:02<02:49,  5.85s/it]
                                                                                                          Rank 0 Step 11: 0.7363902926445007 loss, 5.5559s                            | 11/39 [01:08<02:42,  5.80s/it]
                                                                                                          Rank 0 Step 12: 0.76777583360672 loss, 5.5475s                              | 12/39 [01:14<02:36,  5.78s/it]
                                                                                                          Rank 0 Step 13: 0.735287070274353 loss, 5.5935s                             | 13/39 [01:19<02:29,  5.76s/it]
                                                                                                          Rank 0 Step 14: 0.7306473851203918 loss, 5.5743s                            | 14/39 [01:25<02:24,  5.76s/it]
                                                                                                          Rank 0 Step 15: 0.6343804001808167 loss, 5.5668s                            | 15/39 [01:31<02:18,  5.75s/it]
                                                                                                          Rank 0 Step 16: 0.7133765816688538 loss, 5.5492s                            | 16/39 [01:37<02:12,  5.74s/it]
                                                                                                          Rank 0 Step 17: 0.6251988410949707 loss, 5.5067s                            | 17/39 [01:42<02:06,  5.74s/it]
                                                                                                          Rank 0 Step 18: 0.6921062469482422 loss, 5.5482s                            | 18/39 [01:48<02:00,  5.72s/it]
                                                                                                          Rank 0 Step 19: 0.6669695377349854 loss, 5.5026s                            | 19/39 [01:54<01:54,  5.72s/it]
                                                                                                          Rank 0 Step 20: 0.707284152507782 loss, 5.5959s▎                            | 20/39 [01:59<01:48,  5.70s/it]
                                                                                                          Rank 0 Step 21: 0.6922057867050171 loss, 5.5767s▊                           | 21/39 [02:05<01:43,  5.73s/it]
                                                                                                          Rank 0 Step 22: 0.6656978726387024 loss, 5.5836s██▎                         | 22/39 [02:11<01:37,  5.73s/it]
                                                                                                          Rank 0 Step 23: 0.6616636514663696 loss, 5.4986s███▊                        | 23/39 [02:17<01:31,  5.74s/it]
                                                                                                          Rank 0 Step 24: 0.6690230965614319 loss, 5.6449s█████▎                      | 24/39 [02:22<01:25,  5.72s/it]
                                                                                                          Rank 0 Step 25: 0.6990294456481934 loss, 5.6851s██████▊                     | 25/39 [02:28<01:20,  5.75s/it]
                                                                                                          Rank 0 Step 26: 0.6498890519142151 loss, 5.4792s████████▎                   | 26/39 [02:34<01:15,  5.78s/it]
                                                                                                          Rank 0 Step 27: 0.6175943613052368 loss, 5.5812s█████████▊                  | 27/39 [02:40<01:08,  5.74s/it]
                                                                                                          Rank 0 Step 28: 0.6360677480697632 loss, 5.5406s███████████▎                | 28/39 [02:45<01:03,  5.74s/it]
                                                                                                          Rank 0 Step 29: 0.6467706561088562 loss, 5.5603s████████████▊               | 29/39 [02:51<00:57,  5.73s/it]
                                                                                                          Rank 0 Step 30: 0.7139467000961304 loss, 5.5282s██████████████▍             | 30/39 [02:57<00:51,  5.73s/it]
                                                                                                          Rank 0 Step 31: 0.666486918926239 loss, 5.5621s████████████████▉            | 31/39 [03:02<00:45,  5.72s/it]
                                                                                                          Rank 0 Step 32: 0.6726603507995605 loss, 5.5690s█████████████████▍          | 32/39 [03:08<00:40,  5.73s/it]
                                                                                                          Rank 0 Step 33: 0.6935362815856934 loss, 5.5658s██████████████████▉         | 33/39 [03:14<00:34,  5.73s/it]
                                                                                                          Rank 0 Step 34: 0.6906307935714722 loss, 5.6286s████████████████████▍       | 34/39 [03:20<00:28,  5.73s/it]
                                                                                                          Rank 0 Step 35: 0.6484121680259705 loss, 5.6014s█████████████████████▉      | 35/39 [03:26<00:23,  5.76s/it]
                                                                                                          Rank 0 Step 36: 0.6566431522369385 loss, 5.6330s███████████████████████▍    | 36/39 [03:31<00:17,  5.76s/it]
                                                                                                          Rank 0 Step 37: 0.7162940502166748 loss, 5.5610s████████████████████████▉   | 37/39 [03:37<00:11,  5.77s/it]
                                                                                                          Rank 0 Step 38: 0.6618422865867615 loss, 5.2489s██████████████████████████▍ | 38/39 [03:43<00:05,  5.76s/it]
Iteration: 100%|███████████████████████████████████████████████████████████| 39/39 [03:48<00:00,  5.86s/it]
03/19/2026 15:12:51 - INFO - __main__ -   Loading features from cached file /users/jl0796/COS568-DistLM-SP26/glue_data/RTE/cached_dev_bert-base-cased_128_rte
/users/jl0796/COS568-DistLM-SP26/task2a/run_glue.py:286: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  features = torch.load(cached_features_file)
03/19/2026 15:12:51 - INFO - __main__ -   ***** Running evaluation  *****
03/19/2026 15:12:51 - INFO - __main__ -     Num examples = 277
03/19/2026 15:12:51 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.98it/s]
03/19/2026 15:12:54 - INFO - __main__ -   ***** Eval results  *****██████████| 9/9 [00:03<00:00,  3.14it/s]
03/19/2026 15:12:54 - INFO - __main__ -     acc = 0.5714285714285714
{'acc': np.float64(0.5714285714285714)}
Epoch: 100%|████████████████████████████████████████████████████████████████| 1/1 [03:51<00:00, 231.77s/it]
03/19/2026 15:12:54 - INFO - __main__ -    global_step = 39, average loss = 0.6984662199631716
03/19/2026 15:12:54 - INFO - __main__ -   Loading features from cached file /users/jl0796/COS568-DistLM-SP26/glue_data/RTE/cached_dev_bert-base-cased_128_rte
03/19/2026 15:12:54 - INFO - __main__ -   ***** Running evaluation  *****
03/19/2026 15:12:54 - INFO - __main__ -     Num examples = 277
03/19/2026 15:12:54 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.98it/s]
03/19/2026 15:12:57 - INFO - __main__ -   ***** Eval results  *****
03/19/2026 15:12:57 - INFO - __main__ -     acc = 0.5714285714285714



rank1:
Rank 1 Step 1: 0.8088881969451904 loss, 5.8952s
Rank 1 Step 2: 0.7751753330230713 loss, 6.0130s
Rank 1 Step 3: 0.6154295206069946 loss, 6.0562s
Rank 1 Step 4: 0.7325257062911987 loss, 6.6555s
Rank 1 Step 5: 0.682647705078125 loss, 5.5044s
Rank 1 Step 6: 0.7972484827041626 loss, 5.5218s
Rank 1 Step 7: 0.8497344255447388 loss, 5.5272s
Rank 1 Step 8: 0.7141337394714355 loss, 5.5712s
Rank 1 Step 9: 0.7543344497680664 loss, 5.5471s
Rank 1 Step 10: 0.8139281272888184 loss, 5.5644s
Rank 1 Step 11: 0.744855523109436 loss, 5.5879s
Rank 1 Step 12: 0.7255043983459473 loss, 5.5689s
Rank 1 Step 13: 0.6903994083404541 loss, 5.6209s
Rank 1 Step 14: 0.7242132425308228 loss, 5.5992s
Rank 1 Step 15: 0.7220540642738342 loss, 5.5896s
Rank 1 Step 16: 0.6147739291191101 loss, 5.5587s
Rank 1 Step 17: 0.7293049693107605 loss, 5.5350s
Rank 1 Step 18: 0.692111611366272 loss, 5.5686s
Rank 1 Step 19: 0.6828621625900269 loss, 5.5350s
Rank 1 Step 20: 0.6474546194076538 loss, 5.6244s
Rank 1 Step 21: 0.6743173599243164 loss, 5.6182s
Rank 1 Step 22: 0.6162534356117249 loss, 5.6118s
Rank 1 Step 23: 0.6681494116783142 loss, 5.5247s
Rank 1 Step 24: 0.6844497323036194 loss, 5.6813s
Rank 1 Step 25: 0.6790173053741455 loss, 5.7142s
Rank 1 Step 26: 0.7332025170326233 loss, 5.4941s
Rank 1 Step 27: 0.7437059879302979 loss, 5.6014s
Rank 1 Step 28: 0.7416518926620483 loss, 5.5613s
Rank 1 Step 29: 0.6495735049247742 loss, 5.5851s
Rank 1 Step 30: 0.717158317565918 loss, 5.5527s
Rank 1 Step 31: 0.6885576844215393 loss, 5.6047s
Rank 1 Step 32: 0.6243106126785278 loss, 5.6049s
Rank 1 Step 33: 0.7293746471405029 loss, 5.5910s
Rank 1 Step 34: 0.711456298828125 loss, 5.6608s
Rank 1 Step 35: 0.6508678197860718 loss, 5.6191s
Rank 1 Step 36: 0.679632306098938 loss, 5.6568s
Rank 1 Step 37: 0.6973366141319275 loss, 5.5914s
Rank 1 Step 38: 0.7089684009552002 loss, 5.2674s
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.97it/s]
{'acc': np.float64(0.6714285714285714)}

rank2:
Rank 2 Step 1: 0.6893115043640137 loss, 5.8961s
Rank 2 Step 2: 0.6745952367782593 loss, 6.0327s
Rank 2 Step 3: 0.5807715058326721 loss, 6.0542s
Rank 2 Step 4: 0.6408742666244507 loss, 6.6318s
Rank 2 Step 5: 0.802609384059906 loss, 5.5045s
Rank 2 Step 6: 0.6470295190811157 loss, 5.5236s
Rank 2 Step 7: 0.7225726842880249 loss, 5.5257s
Rank 2 Step 8: 0.7655749917030334 loss, 5.5709s
Rank 2 Step 9: 0.6502664089202881 loss, 5.5476s
Rank 2 Step 10: 0.6749289035797119 loss, 5.5643s
Rank 2 Step 11: 0.656528890132904 loss, 5.5870s
Rank 2 Step 12: 0.6667109727859497 loss, 5.5709s
Rank 2 Step 13: 0.7003048658370972 loss, 5.6225s
Rank 2 Step 14: 0.6759980320930481 loss, 5.5989s
Rank 2 Step 15: 0.7211639881134033 loss, 5.5908s
Rank 2 Step 16: 0.6674907803535461 loss, 5.5585s
Rank 2 Step 17: 0.6738567352294922 loss, 5.5341s
Rank 2 Step 18: 0.6872974038124084 loss, 5.5693s
Rank 2 Step 19: 0.6855839490890503 loss, 5.5320s
Rank 2 Step 20: 0.6864213347434998 loss, 5.6243s
Rank 2 Step 21: 0.7432588338851929 loss, 5.6176s
Rank 2 Step 22: 0.6011860370635986 loss, 5.6114s
Rank 2 Step 23: 0.6793489456176758 loss, 5.5245s
Rank 2 Step 24: 0.604113757610321 loss, 5.6806s
Rank 2 Step 25: 0.7135170102119446 loss, 5.7144s
Rank 2 Step 26: 0.7276502847671509 loss, 5.5077s
Rank 2 Step 27: 0.7308619618415833 loss, 5.6009s
Rank 2 Step 28: 0.6620175838470459 loss, 5.5614s
Rank 2 Step 29: 0.6779232025146484 loss, 5.5857s
Rank 2 Step 30: 0.6656278371810913 loss, 5.5509s
Rank 2 Step 31: 0.643742561340332 loss, 5.5923s
Rank 2 Step 32: 0.6978839039802551 loss, 5.6047s
Rank 2 Step 33: 0.6921669244766235 loss, 5.5906s
Rank 2 Step 34: 0.703972339630127 loss, 5.6583s
Rank 2 Step 35: 0.6654226779937744 loss, 5.6380s
Rank 2 Step 36: 0.678417980670929 loss, 5.6544s
Rank 2 Step 37: 0.7009249925613403 loss, 5.5925s
Rank 2 Step 38: 0.7804261445999146 loss, 5.2787s
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.98it/s]
{'acc': np.float64(0.5714285714285714)}


rank3:
Rank 3 Step 1: 0.6378768682479858 loss, 5.9180s
Rank 3 Step 2: 0.7848365306854248 loss, 6.0467s
Rank 3 Step 3: 0.8473948240280151 loss, 6.0508s
Rank 3 Step 4: 0.7628737092018127 loss, 6.6661s
Rank 3 Step 5: 0.7861862778663635 loss, 5.4976s
Rank 3 Step 6: 0.643917977809906 loss, 5.5269s
Rank 3 Step 7: 0.7529057860374451 loss, 5.5308s
Rank 3 Step 8: 0.625250518321991 loss, 5.5746s
Rank 3 Step 9: 0.6525262594223022 loss, 5.5519s
Rank 3 Step 10: 0.6911729574203491 loss, 5.5677s
Rank 3 Step 11: 0.7579160928726196 loss, 5.5909s
Rank 3 Step 12: 0.7595993876457214 loss, 5.5727s
Rank 3 Step 13: 0.7044041752815247 loss, 5.6255s
Rank 3 Step 14: 0.7202672362327576 loss, 5.5743s
Rank 3 Step 15: 0.6941282749176025 loss, 5.5939s
Rank 3 Step 16: 0.6632963418960571 loss, 5.5611s
Rank 3 Step 17: 0.6638643145561218 loss, 5.5378s
Rank 3 Step 18: 0.7128469347953796 loss, 5.5722s
Rank 3 Step 19: 0.6600683331489563 loss, 5.5366s
Rank 3 Step 20: 0.6472814083099365 loss, 5.6278s
Rank 3 Step 21: 0.6282680034637451 loss, 5.6212s
Rank 3 Step 22: 0.7439930438995361 loss, 5.6157s
Rank 3 Step 23: 0.6477343440055847 loss, 5.5288s
Rank 3 Step 24: 0.6826679706573486 loss, 5.6849s
Rank 3 Step 25: 0.7687021493911743 loss, 5.7177s
Rank 3 Step 26: 0.7249166369438171 loss, 5.5097s
Rank 3 Step 27: 0.6383755207061768 loss, 5.6049s
Rank 3 Step 28: 0.7754285335540771 loss, 5.5646s
Rank 3 Step 29: 0.577017068862915 loss, 5.5894s
Rank 3 Step 30: 0.6482086777687073 loss, 5.5549s
Rank 3 Step 31: 0.6300307512283325 loss, 5.6085s
Rank 3 Step 32: 0.6769802570343018 loss, 5.6085s
Rank 3 Step 33: 0.5858186483383179 loss, 5.5942s
Rank 3 Step 34: 0.677280843257904 loss, 5.6646s
Rank 3 Step 35: 0.642704427242279 loss, 5.6426s
Rank 3 Step 36: 0.6742525100708008 loss, 5.6596s
Rank 3 Step 37: 0.6562798023223877 loss, 5.5960s
Rank 3 Step 38: 0.6848657727241516 loss, 5.2816s
Evaluating: 100%|████████████████████████████████████████████████████████████| 9/9 [00:03<00:00,  2.98it/s]
{'acc': np.float64(0.5571428571428572)}




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
