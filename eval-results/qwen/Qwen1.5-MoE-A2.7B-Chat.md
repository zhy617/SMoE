hf (pretrained=Qwen/Qwen1.5-MoE-A2.7B-Chat,trust_remote_code=True,cache_dir=/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B-Chat), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge                          |      1|none  |     0|acc     |↑  |0.3970|±  |0.0155|
|                                       |       |none  |     0|acc_norm|↑  |0.3980|±  |0.0155|
|arc_easy                               |      1|none  |     0|acc     |↑  |0.7090|±  |0.0144|
|                                       |       |none  |     0|acc_norm|↑  |0.6800|±  |0.0148|
|mmlu                                   |      2|none  |      |acc     |↑  |0.6016|±  |0.0040|
| - humanities                          |      2|none  |      |acc     |↑  |0.5485|±  |0.0071|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.3492|±  |0.0426|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.7212|±  |0.0350|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.7696|±  |0.0296|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.7764|±  |0.0271|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.7521|±  |0.0394|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.7315|±  |0.0428|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.7301|±  |0.0349|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.6445|±  |0.0258|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2726|±  |0.0149|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.6913|±  |0.0262|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.7068|±  |0.0253|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.4480|±  |0.0157|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.7953|±  |0.0309|
| - other                               |      2|none  |      |acc     |↑  |0.6794|±  |0.0081|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.7200|±  |0.0451|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.6868|±  |0.0285|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.5665|±  |0.0378|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.3900|±  |0.0490|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.6054|±  |0.0328|
|  - management                         |      1|none  |     0|acc     |↑  |0.7961|±  |0.0399|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.8590|±  |0.0228|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.6800|±  |0.0469|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.8059|±  |0.0141|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.6830|±  |0.0266|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.4645|±  |0.0298|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.6544|±  |0.0289|
|  - virology                           |      1|none  |     0|acc     |↑  |0.5120|±  |0.0389|
| - social sciences                     |      2|none  |      |acc     |↑  |0.6952|±  |0.0081|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.4035|±  |0.0462|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.8030|±  |0.0283|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.8238|±  |0.0275|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.6077|±  |0.0248|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.6681|±  |0.0306|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.7835|±  |0.0177|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.6794|±  |0.0409|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.5964|±  |0.0198|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.6182|±  |0.0465|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.7469|±  |0.0278|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.8259|±  |0.0268|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.8100|±  |0.0394|
| - stem                                |      2|none  |      |acc     |↑  |0.5040|±  |0.0086|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.2900|±  |0.0456|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.5704|±  |0.0428|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.6382|±  |0.0391|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.6458|±  |0.0400|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.4500|±  |0.0500|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.4700|±  |0.0502|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.3500|±  |0.0479|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.3725|±  |0.0481|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.7200|±  |0.0451|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.5234|±  |0.0327|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.5931|±  |0.0409|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.4153|±  |0.0254|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.7226|±  |0.0255|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.4680|±  |0.0351|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.6300|±  |0.0485|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.3556|±  |0.0292|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.3709|±  |0.0394|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.5185|±  |0.0341|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.3929|±  |0.0464|
|winogrande                             |      1|none  |     0|acc     |↑  |0.6540|±  |0.0151|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6016|±  |0.0040|
| - humanities     |      2|none  |      |acc   |↑  |0.5485|±  |0.0071|
| - other          |      2|none  |      |acc   |↑  |0.6794|±  |0.0081|
| - social sciences|      2|none  |      |acc   |↑  |0.6952|±  |0.0081|
| - stem           |      2|none  |      |acc   |↑  |0.5040|±  |0.0086|