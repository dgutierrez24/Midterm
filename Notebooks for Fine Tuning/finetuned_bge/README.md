---
base_model: BAAI/bge-base-en-v1.5
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
- dot_accuracy@1
- dot_accuracy@3
- dot_accuracy@5
- dot_accuracy@10
- dot_precision@1
- dot_precision@3
- dot_precision@5
- dot_precision@10
- dot_recall@1
- dot_recall@3
- dot_recall@5
- dot_recall@10
- dot_ndcg@10
- dot_mrr@10
- dot_map@100
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:600
- loss:MatryoshkaLoss
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: What is the purpose of the Blueprint for an AI Bill of Rights as
    mentioned in the context?
  sentences:
  - "test its impacts on you \nProportionate. The availability of human consideration\
    \ and fallback, along with associated training and \nsafeguards against human\
    \ bias, should be proportionate to the potential of the automated system to meaning­\n\
    fully impact rights, opportunities, or access. Automated systems that have greater\
    \ control over outcomes, \nprovide input to high-stakes decisions, relate to sensitive\
    \ domains, or otherwise have a greater potential to \nmeaningfully impact rights,\
    \ opportunities, or access should have greater availability (e.g., staffing) and\
    \ over­\nsight of human consideration and fallback mechanisms. \nAccessible. Mechanisms\
    \ for human consideration and fallback, whether in-person, on paper, by phone,\
    \ or"
  - "HUMAN ALTERNATIVES, \nCONSIDERATION, AND \nFALLBACK \nWHY THIS PRINCIPLE IS IMPORTANT\n\
    This section provides a brief summary of the problems which the principle seeks\
    \ to address and protect \nagainst, including illustrative examples. \n•\nAn unemployment\
    \ benefits system in Colorado required, as a condition of accessing benefits,\
    \ that applicants\nhave a smartphone in order to verify their identity. No alternative\
    \ human option was readily available,\nwhich denied many people access to benefits.101\n\
    •\nA fraud detection system for unemployment insurance distribution incorrectly\
    \ flagged entries as fraudulent,\nleading to people with slight discrepancies\
    \ or complexities in their files having their wages withheld and tax"
  - "enforcement, and other regulatory contexts may require government actors to protect\
    \ civil rights, civil liberties, \nand privacy in a manner consistent with, but\
    \ using alternate mechanisms to, the specific principles discussed in \nthis framework.\
    \ The Blueprint for an AI Bill of Rights is meant to assist governments and the\
    \ private sector in \nmoving principles into practice. \nThe expectations given\
    \ in the Technical Companion are meant to serve as a blueprint for the development\
    \ of \nadditional technical standards and practices that should be tailored for\
    \ particular sectors and contexts. While \nexisting laws informed the development\
    \ of the Blueprint for an AI Bill of Rights, this framework does not detail"
- source_sentence: How can user experience research contribute to ensuring that data
    collection practices align with individuals' expectations and desires?
  sentences:
  - "the privacy, civil rights, and civil liberties implications of the use of such\
    \ technologies be issued before \nbiometric identification technologies can be\
    \ used in New York schools. \nFederal law requires employers, and any consultants\
    \ they may retain, to report the costs \nof surveilling employees in the context\
    \ of a labor dispute, providing a transparency \nmechanism to help protect worker\
    \ organizing. Employers engaging in workplace surveillance \"where \nan object\
    \ there-of, directly or indirectly, is […] to obtain information concerning the\
    \ activities of employees or a \nlabor organization in connection with a labor\
    \ dispute\" must report expenditures relating to this surveillance to"
  - "collection should be minimized and clearly communicated to the people whose data\
    \ is collected. Data should \nonly be collected or used for the purposes of training\
    \ or testing machine learning models if such collection and \nuse is legal and\
    \ consistent with the expectations of the people whose data is collected. User\
    \ experience \nresearch should be conducted to confirm that people understand\
    \ what data is being collected about them and \nhow it will be used, and that\
    \ this collection matches their expectations and desires. \nData collection and\
    \ use-case scope limits. Data collection should be limited in scope, with specific,\
    \ \nnarrow identified goals, to avoid \"mission creep.\"  Anticipated data collection\
    \ should be determined to be"
  - "help to mitigate biases and potential harms. \nGuarding against proxies.  Directly\
    \ using demographic information in the design, development, or \ndeployment of\
    \ an automated system (for purposes other than evaluating a system for discrimination\
    \ or using \na system to counter discrimination) runs a high risk of leading to\
    \ algorithmic discrimination and should be \navoided. In many cases, attributes\
    \ that are highly correlated with demographic features, known as proxies, can\
    \ \ncontribute to algorithmic discrimination. In cases where use of the demographic\
    \ features themselves would \nlead to illegal algorithmic discrimination, reliance\
    \ on such proxies in decision-making (such as that facilitated"
- source_sentence: How can surveillance technologies potentially impact your rights,
    opportunities, or access?
  sentences:
  - "enforcement or national security restrictions prevent doing so. Care should be\
    \ taken to balance individual \nprivacy with evaluation data access needs; in\
    \ many cases, policy-based and/or technological innovations and \ncontrols allow\
    \ access to such data without compromising privacy. \nReporting. Entities responsible\
    \ for the development or use of automated systems should provide \nreporting of\
    \ an appropriately designed algorithmic impact assessment,50 with clear specification\
    \ of who \nperforms the assessment, who evaluates the system, and how corrective\
    \ actions are taken (if necessary) in \nresponse to the assessment. This algorithmic\
    \ impact assessment should include at least: the results of any"
  - "SAFE AND EFFECTIVE \nSYSTEMS \nWHY THIS PRINCIPLE IS IMPORTANT\nThis section\
    \ provides a brief summary of the problems which the principle seeks to address\
    \ and protect \nagainst, including illustrative examples. \nWhile technologies\
    \ are being deployed to solve problems across a wide array of issues, our reliance\
    \ on technology can \nalso lead to its use in situations where it has not yet\
    \ been proven to work—either at all or within an acceptable range \nof error.\
    \ In other cases, technologies do not work as intended or as promised, causing\
    \ substantial and unjustified harm. \nAutomated systems sometimes rely on data\
    \ from other systems, including historical data, allowing irrelevant informa­"
  - "access. Whenever possible, you should have access to reporting that confirms\
    \ \nyour data decisions have been respected and provides an assessment of the\
    \ \npotential impact of surveillance technologies on your rights, opportunities,\
    \ or \naccess. \nDATA PRIVACY\n30"
- source_sentence: Why is it difficult for individuals to opt out of sensitive domains
    such as health and employment?
  sentences:
  - "existing human performance considered as a performance baseline for the algorithm\
    \ to meet pre-deployment, \nand as a lifecycle minimum performance standard. Decision\
    \ possibilities resulting from performance testing \nshould include the possibility\
    \ of not deploying the system. \nRisk identification and mitigation. Before deployment,\
    \ and in a proactive and ongoing manner, poten­\ntial risks of the automated system\
    \ should be identified and mitigated. Identified risks should focus on the \n\
    potential for meaningful impact on people’s rights, opportunities, or access and\
    \ include those to impacted \ncommunities that may not be direct users of the\
    \ automated system, risks resulting from purposeful misuse of"
  - "in some cases. Many states have also enacted consumer data privacy protection\
    \ regimes to address some of these \nharms. \nHowever, these are not yet standard\
    \ practices, and the United States lacks a comprehensive statutory or regulatory\
    \ \nframework governing the rights of the public when it comes to personal data.\
    \ While a patchwork of laws exists to \nguide the collection and use of personal\
    \ data in specific contexts, including health, employment, education, and credit,\
    \ \nit can be unclear how these laws apply in other contexts and in an increasingly\
    \ automated society. Additional protec­\ntions would assure the American public\
    \ that the automated systems they use are not monitoring their activities,"
  - "DATA PRIVACY \nEXTRA PROTECTIONS FOR DATA RELATED TO SENSITIVE\nDOMAINS\nSome\
    \ domains, including health, employment, education, criminal justice, and personal\
    \ finance, have long been \nsingled out as sensitive domains deserving of enhanced\
    \ data protections. This is due to the intimate nature of these \ndomains as well\
    \ as the inability of individuals to opt out of these domains in any meaningful\
    \ way, and the \nhistorical discrimination that has often accompanied data knowledge.69\
    \ Domains understood by the public to be \nsensitive also change over time, including\
    \ because of technological developments. Tracking and monitoring \ntechnologies,\
    \ personal tracking devices, and our extensive data footprints are used and misused\
    \ more than ever"
- source_sentence: What are some of the risks associated with using AI in high-stakes
    settings as discussed by the panelists?
  sentences:
  - "(before the technology is built and instituted). Various panelists also emphasized\
    \ the importance of regulation \nthat includes limits to the type and cost of\
    \ such technologies. \n56"
  - "and other data-driven automated systems most directly collect data on, make inferences\
    \ about, and may cause \nharm to individuals. But the overall magnitude of their\
    \ impacts may be most readily visible at the level of com-\nmunities. Accordingly,\
    \ the concept of community is integral to the scope of the Blueprint for an AI\
    \ Bill of Rights. \nUnited States law and policy have long employed approaches\
    \ for protecting the rights of individuals, but exist-\ning frameworks have sometimes\
    \ struggled to provide protections when effects manifest most clearly at a com-\n\
    munity level. For these reasons, the Blueprint for an AI Bill of Rights asserts\
    \ that the harms of automated"
  - "Moderator: Kathy Pham Evans, Deputy Chief Technology Officer for Product and\
    \ Engineering, U.S \nFederal Trade Commission. \nPanelists: \n•\nLiz O’Sullivan,\
    \ CEO, Parity AI\n•\nTimnit Gebru, Independent Scholar\n•\nJennifer Wortman Vaughan,\
    \ Senior Principal Researcher, Microsoft Research, New York City\n•\nPamela Wisniewski,\
    \ Associate Professor of Computer Science, University of Central Florida; Director,\n\
    Socio-technical Interaction Research (STIR) Lab\n•\nSeny Kamara, Associate Professor\
    \ of Computer Science, Brown University\nEach panelist individually emphasized\
    \ the risks of using AI in high-stakes settings, including the potential for \n\
    biased data and discriminatory outcomes, opaque decision-making processes, and\
    \ lack of public trust and"
model-index:
- name: SentenceTransformer based on BAAI/bge-base-en-v1.5
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: cosine_accuracy@1
      value: 0.87
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.97
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 1.0
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 1.0
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.87
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.3233333333333333
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.19999999999999996
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.09999999999999998
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.87
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.97
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 1.0
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 1.0
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.943827499546856
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.9248333333333334
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.9248333333333334
      name: Cosine Map@100
    - type: dot_accuracy@1
      value: 0.87
      name: Dot Accuracy@1
    - type: dot_accuracy@3
      value: 0.97
      name: Dot Accuracy@3
    - type: dot_accuracy@5
      value: 1.0
      name: Dot Accuracy@5
    - type: dot_accuracy@10
      value: 1.0
      name: Dot Accuracy@10
    - type: dot_precision@1
      value: 0.87
      name: Dot Precision@1
    - type: dot_precision@3
      value: 0.3233333333333333
      name: Dot Precision@3
    - type: dot_precision@5
      value: 0.19999999999999996
      name: Dot Precision@5
    - type: dot_precision@10
      value: 0.09999999999999998
      name: Dot Precision@10
    - type: dot_recall@1
      value: 0.87
      name: Dot Recall@1
    - type: dot_recall@3
      value: 0.97
      name: Dot Recall@3
    - type: dot_recall@5
      value: 1.0
      name: Dot Recall@5
    - type: dot_recall@10
      value: 1.0
      name: Dot Recall@10
    - type: dot_ndcg@10
      value: 0.943827499546856
      name: Dot Ndcg@10
    - type: dot_mrr@10
      value: 0.9248333333333334
      name: Dot Mrr@10
    - type: dot_map@100
      value: 0.9248333333333334
      name: Dot Map@100
---

# SentenceTransformer based on BAAI/bge-base-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) <!-- at revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'What are some of the risks associated with using AI in high-stakes settings as discussed by the panelists?',
    'Moderator: Kathy Pham Evans, Deputy Chief Technology Officer for Product and Engineering, U.S \nFederal Trade Commission. \nPanelists: \n•\nLiz O’Sullivan, CEO, Parity AI\n•\nTimnit Gebru, Independent Scholar\n•\nJennifer Wortman Vaughan, Senior Principal Researcher, Microsoft Research, New York City\n•\nPamela Wisniewski, Associate Professor of Computer Science, University of Central Florida; Director,\nSocio-technical Interaction Research (STIR) Lab\n•\nSeny Kamara, Associate Professor of Computer Science, Brown University\nEach panelist individually emphasized the risks of using AI in high-stakes settings, including the potential for \nbiased data and discriminatory outcomes, opaque decision-making processes, and lack of public trust and',
    'and other data-driven automated systems most directly collect data on, make inferences about, and may cause \nharm to individuals. But the overall magnitude of their impacts may be most readily visible at the level of com-\nmunities. Accordingly, the concept of community is integral to the scope of the Blueprint for an AI Bill of Rights. \nUnited States law and policy have long employed approaches for protecting the rights of individuals, but exist-\ning frameworks have sometimes struggled to provide protections when effects manifest most clearly at a com-\nmunity level. For these reasons, the Blueprint for an AI Bill of Rights asserts that the harms of automated',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.87       |
| cosine_accuracy@3   | 0.97       |
| cosine_accuracy@5   | 1.0        |
| cosine_accuracy@10  | 1.0        |
| cosine_precision@1  | 0.87       |
| cosine_precision@3  | 0.3233     |
| cosine_precision@5  | 0.2        |
| cosine_precision@10 | 0.1        |
| cosine_recall@1     | 0.87       |
| cosine_recall@3     | 0.97       |
| cosine_recall@5     | 1.0        |
| cosine_recall@10    | 1.0        |
| cosine_ndcg@10      | 0.9438     |
| cosine_mrr@10       | 0.9248     |
| **cosine_map@100**  | **0.9248** |
| dot_accuracy@1      | 0.87       |
| dot_accuracy@3      | 0.97       |
| dot_accuracy@5      | 1.0        |
| dot_accuracy@10     | 1.0        |
| dot_precision@1     | 0.87       |
| dot_precision@3     | 0.3233     |
| dot_precision@5     | 0.2        |
| dot_precision@10    | 0.1        |
| dot_recall@1        | 0.87       |
| dot_recall@3        | 0.97       |
| dot_recall@5        | 1.0        |
| dot_recall@10       | 1.0        |
| dot_ndcg@10         | 0.9438     |
| dot_mrr@10          | 0.9248     |
| dot_map@100         | 0.9248     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 600 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 600 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              |
  | details | <ul><li>min: 10 tokens</li><li>mean: 19.88 tokens</li><li>max: 36 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 115.61 tokens</li><li>max: 223 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                       | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What is the purpose of the AI Bill of Rights mentioned in the context?</code>                                                              | <code>BLUEPRINT FOR AN <br>AI BILL OF <br>RIGHTS <br>MAKING AUTOMATED <br>SYSTEMS WORK FOR <br>THE AMERICAN PEOPLE <br>OCTOBER 2022</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  | <code>When was the Blueprint for an AI Bill of Rights published?</code>                                                                          | <code>BLUEPRINT FOR AN <br>AI BILL OF <br>RIGHTS <br>MAKING AUTOMATED <br>SYSTEMS WORK FOR <br>THE AMERICAN PEOPLE <br>OCTOBER 2022</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  | <code>What is the purpose of the Blueprint for an AI Bill of Rights published by the White House Office of Science and Technology Policy?</code> | <code>About this Document <br>The Blueprint for an AI Bill of Rights: Making Automated Systems Work for the American People was <br>published by the White House Office of Science and Technology Policy in October 2022. This framework was <br>released one year after OSTP announced the launch of a process to develop “a bill of rights for an AI-powered <br>world.” Its release follows a year of public engagement to inform this initiative. The framework is available <br>online at: https://www.whitehouse.gov/ostp/ai-bill-of-rights <br>About the Office of Science and Technology Policy <br>The Office of Science and Technology Policy (OSTP) was established by the National Science and Technology</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          768,
          512,
          256,
          128,
          64
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 20
- `per_device_eval_batch_size`: 20
- `num_train_epochs`: 5
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 20
- `per_device_eval_batch_size`: 20
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | cosine_map@100 |
|:------:|:----:|:--------------:|
| 1.0    | 30   | 0.92           |
| 1.6667 | 50   | 0.9248         |


### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.1.1
- Transformers: 4.44.2
- PyTorch: 2.4.1+cu121
- Accelerate: 0.34.2
- Datasets: 3.0.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MatryoshkaLoss
```bibtex
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->