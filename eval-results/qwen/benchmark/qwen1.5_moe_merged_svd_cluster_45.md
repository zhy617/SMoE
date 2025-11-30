hf (pretrained=/root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_cluster_45,trust_remote_code=True), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge                          |      1|none  |     0|acc     |↑  |0.3400|±  |0.0150|
|                                       |       |none  |     0|acc_norm|↑  |0.3850|±  |0.0154|
|arc_easy                               |      1|none  |     0|acc     |↑  |0.6110|±  |0.0154|
|                                       |       |none  |     0|acc_norm|↑  |0.5920|±  |0.0155|
|mmlu                                   |      2|none  |      |acc     |↑  |0.3884|±  |0.0041|
| - humanities                          |      2|none  |      |acc     |↑  |0.3524|±  |0.0073|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.2857|±  |0.0404|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.4545|±  |0.0389|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.4412|±  |0.0348|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.4473|±  |0.0324|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.5620|±  |0.0453|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.3796|±  |0.0469|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.3865|±  |0.0383|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.3353|±  |0.0254|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2469|±  |0.0144|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.4212|±  |0.0280|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.4259|±  |0.0275|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.2860|±  |0.0143|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.5789|±  |0.0379|
| - other                               |      2|none  |      |acc     |↑  |0.4500|±  |0.0088|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.4200|±  |0.0496|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.4113|±  |0.0303|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.3931|±  |0.0372|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.3400|±  |0.0476|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.3094|±  |0.0310|
|  - management                         |      1|none  |     0|acc     |↑  |0.5825|±  |0.0488|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.5513|±  |0.0326|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.4300|±  |0.0498|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.5632|±  |0.0177|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.4477|±  |0.0285|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.2872|±  |0.0270|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.4559|±  |0.0303|
|  - virology                           |      1|none  |     0|acc     |↑  |0.3675|±  |0.0375|
| - social sciences                     |      2|none  |      |acc     |↑  |0.4345|±  |0.0088|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.2368|±  |0.0400|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.5455|±  |0.0355|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.4870|±  |0.0361|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.3846|±  |0.0247|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.4496|±  |0.0323|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.5450|±  |0.0214|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.4427|±  |0.0436|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.3219|±  |0.0189|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.4091|±  |0.0471|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.4449|±  |0.0318|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.4627|±  |0.0353|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.5200|±  |0.0502|
| - stem                                |      2|none  |      |acc     |↑  |0.3302|±  |0.0083|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.2100|±  |0.0409|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.4148|±  |0.0426|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.4211|±  |0.0402|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.4097|±  |0.0411|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.3800|±  |0.0488|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.2800|±  |0.0451|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.2800|±  |0.0451|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.2647|±  |0.0439|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.4400|±  |0.0499|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.3021|±  |0.0300|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.3655|±  |0.0401|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.2460|±  |0.0222|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.4677|±  |0.0284|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.2709|±  |0.0313|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.3800|±  |0.0488|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.2815|±  |0.0274|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.3377|±  |0.0386|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.2917|±  |0.0310|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.2768|±  |0.0425|
|winogrande                             |      1|none  |     0|acc     |↑  |0.6100|±  |0.0154|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.3884|±  |0.0041|
| - humanities     |      2|none  |      |acc   |↑  |0.3524|±  |0.0073|
| - other          |      2|none  |      |acc   |↑  |0.4500|±  |0.0088|
| - social sciences|      2|none  |      |acc   |↑  |0.4345|±  |0.0088|
| - stem           |      2|none  |      |acc   |↑  |0.3302|±  |0.0083|