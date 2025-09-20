hf (pretrained=/root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_cluster_30,trust_remote_code=True), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge                          |      1|none  |     0|acc     |↑  |0.2650|±  |0.0140|
|                                       |       |none  |     0|acc_norm|↑  |0.2930|±  |0.0144|
|arc_easy                               |      1|none  |     0|acc     |↑  |0.5090|±  |0.0158|
|                                       |       |none  |     0|acc_norm|↑  |0.4880|±  |0.0158|
|mmlu                                   |      2|none  |      |acc     |↑  |0.2305|±  |0.0036|
| - humanities                          |      2|none  |      |acc     |↑  |0.2402|±  |0.0066|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.2698|±  |0.0397|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.2242|±  |0.0326|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.2500|±  |0.0304|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.2827|±  |0.0293|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.2397|±  |0.0390|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.2685|±  |0.0428|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.2086|±  |0.0319|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.2486|±  |0.0233|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2380|±  |0.0142|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.1801|±  |0.0218|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.2160|±  |0.0229|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.2460|±  |0.0136|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.2924|±  |0.0349|
| - other                               |      2|none  |      |acc     |↑  |0.2449|±  |0.0077|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.3000|±  |0.0461|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.2226|±  |0.0256|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.2254|±  |0.0319|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.1900|±  |0.0394|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.3094|±  |0.0310|
|  - management                         |      1|none  |     0|acc     |↑  |0.1748|±  |0.0376|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.2949|±  |0.0299|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.3100|±  |0.0465|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.2465|±  |0.0154|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.2190|±  |0.0237|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.2340|±  |0.0253|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.1875|±  |0.0237|
|  - virology                           |      1|none  |     0|acc     |↑  |0.3012|±  |0.0357|
| - social sciences                     |      2|none  |      |acc     |↑  |0.2194|±  |0.0075|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.2368|±  |0.0400|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.1818|±  |0.0275|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.1917|±  |0.0284|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.2026|±  |0.0204|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.2101|±  |0.0265|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.2018|±  |0.0172|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.2595|±  |0.0384|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.2516|±  |0.0176|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.2182|±  |0.0396|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.1878|±  |0.0250|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.2488|±  |0.0306|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.2800|±  |0.0451|
| - stem                                |      2|none  |      |acc     |↑  |0.2144|±  |0.0073|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.2100|±  |0.0409|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.2074|±  |0.0350|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.1908|±  |0.0320|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.2708|±  |0.0372|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.2000|±  |0.0402|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.2600|±  |0.0441|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.2100|±  |0.0409|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.2157|±  |0.0409|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.2700|±  |0.0446|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.2638|±  |0.0288|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.2414|±  |0.0357|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.2116|±  |0.0210|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.1806|±  |0.0219|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.1675|±  |0.0263|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.2200|±  |0.0416|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.2111|±  |0.0249|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.1788|±  |0.0313|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.1574|±  |0.0248|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.3214|±  |0.0443|
|winogrande                             |      1|none  |     0|acc     |↑  |0.5390|±  |0.0158|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.2305|±  |0.0036|
| - humanities     |      2|none  |      |acc   |↑  |0.2402|±  |0.0066|
| - other          |      2|none  |      |acc   |↑  |0.2449|±  |0.0077|
| - social sciences|      2|none  |      |acc   |↑  |0.2194|±  |0.0075|
| - stem           |      2|none  |      |acc   |↑  |0.2144|±  |0.0073|