hf (pretrained=/root/fsas/zhanghongyu/SMoE/qwen/finetuned_models/qwen1.5_moe_merged_svd_cluster_45/final_model,trust_remote_code=True), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge                          |      1|none  |     0|acc     |↑  |0.3960|±  |0.0155|
|                                       |       |none  |     0|acc_norm|↑  |0.3950|±  |0.0155|
|arc_easy                               |      1|none  |     0|acc     |↑  |0.7050|±  |0.0144|
|                                       |       |none  |     0|acc_norm|↑  |0.6740|±  |0.0148|
|boolq                                  |      2|none  |     0|acc     |↑  |0.7480|±  |0.0137|
|hellaswag                              |      1|none  |     0|acc     |↑  |0.4820|±  |0.0158|
|                                       |       |none  |     0|acc_norm|↑  |0.6130|±  |0.0154|
|mmlu                                   |      2|none  |      |acc     |↑  |0.4965|±  |0.0041|
| - humanities                          |      2|none  |      |acc     |↑  |0.4435|±  |0.0073|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.3016|±  |0.0410|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.6242|±  |0.0378|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.6078|±  |0.0343|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.6667|±  |0.0307|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.5868|±  |0.0450|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.5370|±  |0.0482|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.5031|±  |0.0393|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.5491|±  |0.0268|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2469|±  |0.0144|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.5466|±  |0.0283|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.5432|±  |0.0277|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.3400|±  |0.0150|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.6959|±  |0.0353|
| - other                               |      2|none  |      |acc     |↑  |0.5687|±  |0.0087|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.5900|±  |0.0494|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.5660|±  |0.0305|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.4913|±  |0.0381|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.3100|±  |0.0465|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.5426|±  |0.0334|
|  - management                         |      1|none  |     0|acc     |↑  |0.6893|±  |0.0458|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.7521|±  |0.0283|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.4900|±  |0.0502|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.6603|±  |0.0169|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.5654|±  |0.0284|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.4113|±  |0.0294|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.5221|±  |0.0303|
|  - virology                           |      1|none  |     0|acc     |↑  |0.4639|±  |0.0388|
| - social sciences                     |      2|none  |      |acc     |↑  |0.5707|±  |0.0087|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.3421|±  |0.0446|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.7222|±  |0.0319|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.6684|±  |0.0340|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.4744|±  |0.0253|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.4874|±  |0.0325|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.6514|±  |0.0204|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.6260|±  |0.0424|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.4722|±  |0.0202|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.5636|±  |0.0475|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.5918|±  |0.0315|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.7015|±  |0.0324|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.7000|±  |0.0461|
| - stem                                |      2|none  |      |acc     |↑  |0.4231|±  |0.0087|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.3100|±  |0.0465|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.4889|±  |0.0432|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.4934|±  |0.0407|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.5000|±  |0.0418|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.3900|±  |0.0490|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.3400|±  |0.0476|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.3000|±  |0.0461|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.3235|±  |0.0466|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.6400|±  |0.0482|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.4723|±  |0.0326|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.5310|±  |0.0416|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.3360|±  |0.0243|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.5806|±  |0.0281|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.3793|±  |0.0341|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.5100|±  |0.0502|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.3407|±  |0.0289|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.3245|±  |0.0382|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.4120|±  |0.0336|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.3304|±  |0.0446|
|rte                                    |      1|none  |     0|acc     |↑  |0.6209|±  |0.0292|
|winogrande                             |      1|none  |     0|acc     |↑  |0.6520|±  |0.0151|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.4965|±  |0.0041|
| - humanities     |      2|none  |      |acc   |↑  |0.4435|±  |0.0073|
| - other          |      2|none  |      |acc   |↑  |0.5687|±  |0.0087|
| - social sciences|      2|none  |      |acc   |↑  |0.5707|±  |0.0087|
| - stem           |      2|none  |      |acc   |↑  |0.4231|±  |0.0087|