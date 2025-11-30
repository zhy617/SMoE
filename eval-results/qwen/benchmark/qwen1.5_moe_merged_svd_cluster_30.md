hf (pretrained=/root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_cluster_30,trust_remote_code=True), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge                          |      1|none  |     0|acc     |↑  |0.2390|±  |0.0135|
|                                       |       |none  |     0|acc_norm|↑  |0.2560|±  |0.0138|
|arc_easy                               |      1|none  |     0|acc     |↑  |0.4320|±  |0.0157|
|                                       |       |none  |     0|acc_norm|↑  |0.4130|±  |0.0156|
|mmlu                                   |      2|none  |      |acc     |↑  |0.2331|±  |0.0036|
| - humanities                          |      2|none  |      |acc     |↑  |0.2445|±  |0.0067|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.2937|±  |0.0407|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.2242|±  |0.0326|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.2500|±  |0.0304|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.2785|±  |0.0292|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.2893|±  |0.0414|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.2315|±  |0.0408|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.2270|±  |0.0329|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.2486|±  |0.0233|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2380|±  |0.0142|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.2122|±  |0.0232|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.2099|±  |0.0227|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.2450|±  |0.0136|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.3158|±  |0.0357|
| - other                               |      2|none  |      |acc     |↑  |0.2359|±  |0.0076|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.2900|±  |0.0456|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.2302|±  |0.0259|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.2197|±  |0.0316|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.2100|±  |0.0409|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.3004|±  |0.0308|
|  - management                         |      1|none  |     0|acc     |↑  |0.1748|±  |0.0376|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.3034|±  |0.0301|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.2700|±  |0.0446|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.2324|±  |0.0151|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.1895|±  |0.0224|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.2270|±  |0.0250|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.1838|±  |0.0235|
|  - virology                           |      1|none  |     0|acc     |↑  |0.2831|±  |0.0351|
| - social sciences                     |      2|none  |      |acc     |↑  |0.2246|±  |0.0075|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.2368|±  |0.0400|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.2071|±  |0.0289|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.2021|±  |0.0290|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.1718|±  |0.0191|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.2437|±  |0.0279|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.1982|±  |0.0171|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.2519|±  |0.0381|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.2598|±  |0.0177|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.2000|±  |0.0383|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.2163|±  |0.0264|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.2786|±  |0.0317|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.2800|±  |0.0451|
| - stem                                |      2|none  |      |acc     |↑  |0.2236|±  |0.0074|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.2200|±  |0.0416|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.2074|±  |0.0350|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.1908|±  |0.0320|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.2847|±  |0.0377|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.2000|±  |0.0402|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.2500|±  |0.0435|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.2200|±  |0.0416|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.2255|±  |0.0416|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.3100|±  |0.0465|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.2681|±  |0.0290|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.2414|±  |0.0357|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.2063|±  |0.0208|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.1903|±  |0.0223|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.2069|±  |0.0285|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.2800|±  |0.0451|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.2296|±  |0.0256|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.1788|±  |0.0313|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.1667|±  |0.0254|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.3036|±  |0.0436|
|winogrande                             |      1|none  |     0|acc     |↑  |0.5280|±  |0.0158|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.2331|±  |0.0036|
| - humanities     |      2|none  |      |acc   |↑  |0.2445|±  |0.0067|
| - other          |      2|none  |      |acc   |↑  |0.2359|±  |0.0076|
| - social sciences|      2|none  |      |acc   |↑  |0.2246|±  |0.0075|
| - stem           |      2|none  |      |acc   |↑  |0.2236|±  |0.0074|