hf (pretrained=/root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_cluster_45,trust_remote_code=True), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge                          |      1|none  |     0|acc     |↑  |0.3870|±  |0.0154|
|                                       |       |none  |     0|acc_norm|↑  |0.4000|±  |0.0155|
|arc_easy                               |      1|none  |     0|acc     |↑  |0.6840|±  |0.0147|
|                                       |       |none  |     0|acc_norm|↑  |0.6760|±  |0.0148|
|mmlu                                   |      2|none  |      |acc     |↑  |0.4778|±  |0.0042|
| - humanities                          |      2|none  |      |acc     |↑  |0.4280|±  |0.0074|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.3413|±  |0.0424|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.6061|±  |0.0382|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.5833|±  |0.0346|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.6245|±  |0.0315|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.5702|±  |0.0452|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.5185|±  |0.0483|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.4601|±  |0.0392|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.5318|±  |0.0269|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2469|±  |0.0144|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.5016|±  |0.0284|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.5370|±  |0.0277|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.3320|±  |0.0149|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.6316|±  |0.0370|
| - other                               |      2|none  |      |acc     |↑  |0.5401|±  |0.0087|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.5200|±  |0.0502|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.5321|±  |0.0307|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.4566|±  |0.0380|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.3000|±  |0.0461|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.5247|±  |0.0335|
|  - management                         |      1|none  |     0|acc     |↑  |0.6602|±  |0.0469|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.7479|±  |0.0284|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.4300|±  |0.0498|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.6437|±  |0.0171|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.5425|±  |0.0285|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.3511|±  |0.0285|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.4743|±  |0.0303|
|  - virology                           |      1|none  |     0|acc     |↑  |0.4518|±  |0.0387|
| - social sciences                     |      2|none  |      |acc     |↑  |0.5492|±  |0.0088|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.3246|±  |0.0440|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.6717|±  |0.0335|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.6218|±  |0.0350|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.4692|±  |0.0253|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.4454|±  |0.0323|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.6404|±  |0.0206|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.6107|±  |0.0428|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.4542|±  |0.0201|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.5455|±  |0.0477|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.5878|±  |0.0315|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.6617|±  |0.0335|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.6700|±  |0.0473|
| - stem                                |      2|none  |      |acc     |↑  |0.4126|±  |0.0086|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.2500|±  |0.0435|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.5111|±  |0.0432|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.4671|±  |0.0406|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.5208|±  |0.0418|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.3700|±  |0.0485|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.3200|±  |0.0469|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.3500|±  |0.0479|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.2843|±  |0.0449|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.5800|±  |0.0496|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.4681|±  |0.0326|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.5379|±  |0.0415|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.3280|±  |0.0242|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.5387|±  |0.0284|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.3793|±  |0.0341|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.4600|±  |0.0501|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.3667|±  |0.0294|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.3113|±  |0.0378|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.4167|±  |0.0336|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.2857|±  |0.0429|
|winogrande                             |      1|none  |     0|acc     |↑  |0.6180|±  |0.0154|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.4778|±  |0.0042|
| - humanities     |      2|none  |      |acc   |↑  |0.4280|±  |0.0074|
| - other          |      2|none  |      |acc   |↑  |0.5401|±  |0.0087|
| - social sciences|      2|none  |      |acc   |↑  |0.5492|±  |0.0088|
| - stem           |      2|none  |      |acc   |↑  |0.4126|±  |0.0086|