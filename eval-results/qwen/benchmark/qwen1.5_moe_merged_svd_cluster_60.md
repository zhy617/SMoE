hf (pretrained=/root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_cluster_60,trust_remote_code=True), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge                          |      1|none  |     0|acc     |↑  |0.4240|±  |0.0156|
|                                       |       |none  |     0|acc_norm|↑  |0.4330|±  |0.0157|
|arc_easy                               |      1|none  |     0|acc     |↑  |0.7380|±  |0.0139|
|                                       |       |none  |     0|acc_norm|↑  |0.6860|±  |0.0147|
|mmlu                                   |      2|none  |      |acc     |↑  |0.6147|±  |0.0040|
| - humanities                          |      2|none  |      |acc     |↑  |0.5622|±  |0.0071|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.3254|±  |0.0419|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.6909|±  |0.0361|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.7745|±  |0.0293|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.7975|±  |0.0262|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.7769|±  |0.0380|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.7315|±  |0.0428|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.7178|±  |0.0354|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.6676|±  |0.0254|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2994|±  |0.0153|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.6752|±  |0.0266|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.7191|±  |0.0250|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.4710|±  |0.0158|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.8187|±  |0.0295|
| - other                               |      2|none  |      |acc     |↑  |0.6881|±  |0.0080|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.7100|±  |0.0456|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.6717|±  |0.0289|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.6069|±  |0.0372|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.3500|±  |0.0479|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.6323|±  |0.0324|
|  - management                         |      1|none  |     0|acc     |↑  |0.8350|±  |0.0368|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.8547|±  |0.0231|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.6800|±  |0.0469|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.8199|±  |0.0137|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.6863|±  |0.0266|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.4858|±  |0.0298|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.6544|±  |0.0289|
|  - virology                           |      1|none  |     0|acc     |↑  |0.5241|±  |0.0389|
| - social sciences                     |      2|none  |      |acc     |↑  |0.7023|±  |0.0080|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.3509|±  |0.0449|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.7778|±  |0.0296|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.8083|±  |0.0284|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.5974|±  |0.0249|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.6681|±  |0.0306|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.8183|±  |0.0165|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.6947|±  |0.0404|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.6144|±  |0.0197|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.6364|±  |0.0461|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.7592|±  |0.0274|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.8358|±  |0.0262|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.8200|±  |0.0386|
| - stem                                |      2|none  |      |acc     |↑  |0.5262|±  |0.0087|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.4100|±  |0.0494|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.5407|±  |0.0430|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.6250|±  |0.0394|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.6736|±  |0.0392|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.4200|±  |0.0496|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.4200|±  |0.0496|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.4400|±  |0.0499|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.4314|±  |0.0493|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.7500|±  |0.0435|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.5532|±  |0.0325|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.5931|±  |0.0409|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.4524|±  |0.0256|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.7387|±  |0.0250|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.5222|±  |0.0351|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.6100|±  |0.0490|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.4222|±  |0.0301|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.4172|±  |0.0403|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.4722|±  |0.0340|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.3929|±  |0.0464|
|winogrande                             |      1|none  |     0|acc     |↑  |0.6940|±  |0.0146|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6147|±  |0.0040|
| - humanities     |      2|none  |      |acc   |↑  |0.5622|±  |0.0071|
| - other          |      2|none  |      |acc   |↑  |0.6881|±  |0.0080|
| - social sciences|      2|none  |      |acc   |↑  |0.7023|±  |0.0080|
| - stem           |      2|none  |      |acc   |↑  |0.5262|±  |0.0087|