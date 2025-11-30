hf (pretrained=Qwen/Qwen1.5-MoE-A2.7B,trust_remote_code=True,cache_dir=/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge                          |      1|none  |     0|acc     |↑  |0.4220|±  |0.0156|
|                                       |       |none  |     0|acc_norm|↑  |0.4390|±  |0.0157|
|arc_easy                               |      1|none  |     0|acc     |↑  |0.7380|±  |0.0139|
|                                       |       |none  |     0|acc_norm|↑  |0.6850|±  |0.0147|
|mmlu                                   |      2|none  |      |acc     |↑  |0.6153|±  |0.0040|
| - humanities                          |      2|none  |      |acc     |↑  |0.5641|±  |0.0071|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.3175|±  |0.0416|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.7030|±  |0.0357|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.7696|±  |0.0296|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.7890|±  |0.0266|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.7769|±  |0.0380|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.7315|±  |0.0428|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.7239|±  |0.0351|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.6792|±  |0.0251|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.3084|±  |0.0154|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.6752|±  |0.0266|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.7160|±  |0.0251|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.4680|±  |0.0158|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.8246|±  |0.0292|
| - other                               |      2|none  |      |acc     |↑  |0.6852|±  |0.0080|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.7000|±  |0.0461|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.6717|±  |0.0289|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.6012|±  |0.0373|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.3600|±  |0.0482|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.6413|±  |0.0322|
|  - management                         |      1|none  |     0|acc     |↑  |0.7961|±  |0.0399|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.8590|±  |0.0228|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.6700|±  |0.0473|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.8148|±  |0.0139|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.6895|±  |0.0265|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.4823|±  |0.0298|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.6471|±  |0.0290|
|  - virology                           |      1|none  |     0|acc     |↑  |0.5241|±  |0.0389|
| - social sciences                     |      2|none  |      |acc     |↑  |0.7023|±  |0.0080|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.3421|±  |0.0446|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.7980|±  |0.0286|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.8031|±  |0.0287|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.5949|±  |0.0249|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.6639|±  |0.0307|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.8110|±  |0.0168|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.6870|±  |0.0407|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.6160|±  |0.0197|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.6545|±  |0.0455|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.7633|±  |0.0272|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.8408|±  |0.0259|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.8200|±  |0.0386|
| - stem                                |      2|none  |      |acc     |↑  |0.5293|±  |0.0087|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.4000|±  |0.0492|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.5481|±  |0.0430|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.6316|±  |0.0393|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.6736|±  |0.0392|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.4300|±  |0.0498|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.4300|±  |0.0498|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.4300|±  |0.0498|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.4510|±  |0.0495|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.7400|±  |0.0441|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.5532|±  |0.0325|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.5931|±  |0.0409|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.4603|±  |0.0257|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.7355|±  |0.0251|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.5123|±  |0.0352|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.6100|±  |0.0490|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.4370|±  |0.0302|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.4238|±  |0.0403|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.4815|±  |0.0341|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.3929|±  |0.0464|
|winogrande                             |      1|none  |     0|acc     |↑  |0.6990|±  |0.0145|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6153|±  |0.0040|
| - humanities     |      2|none  |      |acc   |↑  |0.5641|±  |0.0071|
| - other          |      2|none  |      |acc   |↑  |0.6852|±  |0.0080|
| - social sciences|      2|none  |      |acc   |↑  |0.7023|±  |0.0080|
| - stem           |      2|none  |      |acc   |↑  |0.5293|±  |0.0087|