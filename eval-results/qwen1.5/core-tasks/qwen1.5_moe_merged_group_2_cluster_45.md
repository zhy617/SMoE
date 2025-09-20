hf (pretrained=/root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_cluster_45,trust_remote_code=True), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge                          |      1|none  |     0|acc     |↑  |0.3540|±  |0.0151|
|                                       |       |none  |     0|acc_norm|↑  |0.3780|±  |0.0153|
|arc_easy                               |      1|none  |     0|acc     |↑  |0.6410|±  |0.0152|
|                                       |       |none  |     0|acc_norm|↑  |0.6210|±  |0.0153|
|mmlu                                   |      2|none  |      |acc     |↑  |0.3588|±  |0.0041|
| - humanities                          |      2|none  |      |acc     |↑  |0.3397|±  |0.0073|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.2222|±  |0.0372|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.4182|±  |0.0385|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.3725|±  |0.0339|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.4304|±  |0.0322|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.4132|±  |0.0450|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.3889|±  |0.0471|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.3313|±  |0.0370|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.3584|±  |0.0258|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2603|±  |0.0147|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.3698|±  |0.0274|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.4383|±  |0.0276|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.3040|±  |0.0146|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.4561|±  |0.0382|
| - other                               |      2|none  |      |acc     |↑  |0.4078|±  |0.0087|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.4700|±  |0.0502|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.3698|±  |0.0297|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.2832|±  |0.0344|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.2700|±  |0.0446|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.4664|±  |0.0335|
|  - management                         |      1|none  |     0|acc     |↑  |0.4757|±  |0.0494|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.4744|±  |0.0327|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.3800|±  |0.0488|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.5070|±  |0.0179|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.4052|±  |0.0281|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.2340|±  |0.0253|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.3346|±  |0.0287|
|  - virology                           |      1|none  |     0|acc     |↑  |0.3976|±  |0.0381|
| - social sciences                     |      2|none  |      |acc     |↑  |0.3893|±  |0.0087|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.2105|±  |0.0384|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.4899|±  |0.0356|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.3731|±  |0.0349|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.2897|±  |0.0230|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.3235|±  |0.0304|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.4128|±  |0.0211|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.4809|±  |0.0438|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.3529|±  |0.0193|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.4455|±  |0.0476|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.4571|±  |0.0319|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.4826|±  |0.0353|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.5300|±  |0.0502|
| - stem                                |      2|none  |      |acc     |↑  |0.3061|±  |0.0082|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.2200|±  |0.0416|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.3630|±  |0.0415|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.3684|±  |0.0393|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.3889|±  |0.0408|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.2200|±  |0.0416|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.3100|±  |0.0465|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.2200|±  |0.0416|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.2549|±  |0.0434|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.4400|±  |0.0499|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.3702|±  |0.0316|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.4207|±  |0.0411|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.2910|±  |0.0234|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.3258|±  |0.0267|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.2167|±  |0.0290|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.3500|±  |0.0479|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.2704|±  |0.0271|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.2848|±  |0.0368|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.2222|±  |0.0284|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.3125|±  |0.0440|
|winogrande                             |      1|none  |     0|acc     |↑  |0.6040|±  |0.0155|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.3588|±  |0.0041|
| - humanities     |      2|none  |      |acc   |↑  |0.3397|±  |0.0073|
| - other          |      2|none  |      |acc   |↑  |0.4078|±  |0.0087|
| - social sciences|      2|none  |      |acc   |↑  |0.3893|±  |0.0087|
| - stem           |      2|none  |      |acc   |↑  |0.3061|±  |0.0082|