hf (pretrained=/root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_cluster_30,trust_remote_code=True), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge                          |      1|none  |     0|acc     |↑  |0.2200|±  |0.0131|
|                                       |       |none  |     0|acc_norm|↑  |0.2400|±  |0.0135|
|arc_easy                               |      1|none  |     0|acc     |↑  |0.3280|±  |0.0149|
|                                       |       |none  |     0|acc_norm|↑  |0.3460|±  |0.0151|
|mmlu                                   |      2|none  |      |acc     |↑  |0.2288|±  |0.0036|
| - humanities                          |      2|none  |      |acc     |↑  |0.2414|±  |0.0066|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.2778|±  |0.0401|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.2182|±  |0.0323|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.2500|±  |0.0304|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.2700|±  |0.0289|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.2397|±  |0.0390|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.2593|±  |0.0424|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.2209|±  |0.0326|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.2486|±  |0.0233|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2380|±  |0.0142|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.1865|±  |0.0221|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.2160|±  |0.0229|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.2460|±  |0.0136|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.3216|±  |0.0358|
| - other                               |      2|none  |      |acc     |↑  |0.2398|±  |0.0076|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.3000|±  |0.0461|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.2151|±  |0.0253|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.2081|±  |0.0310|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.1800|±  |0.0386|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.3139|±  |0.0311|
|  - management                         |      1|none  |     0|acc     |↑  |0.1748|±  |0.0376|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.2906|±  |0.0297|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.3000|±  |0.0461|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.2375|±  |0.0152|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.2255|±  |0.0239|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.2340|±  |0.0253|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.1838|±  |0.0235|
|  - virology                           |      1|none  |     0|acc     |↑  |0.2831|±  |0.0351|
| - social sciences                     |      2|none  |      |acc     |↑  |0.2171|±  |0.0074|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.2368|±  |0.0400|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.1768|±  |0.0272|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.1969|±  |0.0287|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.2026|±  |0.0204|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.2101|±  |0.0265|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.1927|±  |0.0169|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.2595|±  |0.0384|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.2500|±  |0.0175|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.2182|±  |0.0396|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.1878|±  |0.0250|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.2438|±  |0.0304|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.2800|±  |0.0451|
| - stem                                |      2|none  |      |acc     |↑  |0.2125|±  |0.0073|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.2200|±  |0.0416|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.1852|±  |0.0336|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.1776|±  |0.0311|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.2569|±  |0.0365|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.2000|±  |0.0402|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.2600|±  |0.0441|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.2100|±  |0.0409|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.2157|±  |0.0409|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.2800|±  |0.0451|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.2638|±  |0.0288|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.2414|±  |0.0357|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.2090|±  |0.0209|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.1774|±  |0.0217|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.1527|±  |0.0253|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.2500|±  |0.0435|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.2111|±  |0.0249|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.1987|±  |0.0326|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.1528|±  |0.0245|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.3125|±  |0.0440|
|winogrande                             |      1|none  |     0|acc     |↑  |0.5130|±  |0.0158|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.2288|±  |0.0036|
| - humanities     |      2|none  |      |acc   |↑  |0.2414|±  |0.0066|
| - other          |      2|none  |      |acc   |↑  |0.2398|±  |0.0076|
| - social sciences|      2|none  |      |acc   |↑  |0.2171|±  |0.0074|
| - stem           |      2|none  |      |acc   |↑  |0.2125|±  |0.0073|