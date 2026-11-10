# TRL format check

- TRL version: `1.17.1`; documentation checked: <https://huggingface.co/docs/trl/v1.17.1/en/dataset_formats> (DPOTrainer expects a *preference* dataset, explicit prompt recommended; conversational form uses `prompt` = list of messages, `chosen`/`rejected` = list with the assistant message).
- Batch: first 8 rows of `data/manifests/smoke/E1_random.jsonl`; tokenizer `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4c`.
- Collated batch tensor shapes: `{'input_ids': [8, 847], 'attention_mask': [8, 847], 'completion_mask': [8, 847]}`; one DPO step loss = 0.6931 (tiny random-weight model; plumbing check only).

| check | passed |
|---|---|
| is_conversational | True |
| matches_trl_extract_prompt | True |
| no_user_turn_in_completions | True |
| user_turns_rendered_once | True |
| prompt_is_prefix_of_completions | True |
| trl_keeps_full_prompt | True |
| trl_completion_tokens_exact | True |
| real_batch_step_finite_loss | True |

## First example as tokenized by DPOTrainer

**prompt_ids decoded:**

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Determine the degree of resemblance between a feline mammal with sharp retractable claws and a canine mammal with pointed ears and a bushy tail. Specifically, analyze the physical, behavioral, and genetic attributes of both animals to ascertain the level of similarity between them. The two objects in question are commonly known as "cat" and "dog".<|im_end|>
<|im_start|>assistant

```

**chosen_ids decoded (first 400 chars):**

```text
Hello! I'm here to help you with your question about the similarities between cats and dogs. Let's analyze the physical, behavioral, and genetic attributes of both animals step by step to determine the degree of resemblance between them.

1. Physical Attributes: Cats and dogs are alike in some ways, but they also have distinct physical features. Cats are generally smaller in size, with slender bod
```

**rejected_ids decoded (first 400 chars):**

```text
Cats and dogs are both popular household pets and belong to the class of mammals, but they have a number of differences in terms of physical characteristics, behavior, and genetics.

Physical Attributes:
- Body Structure: Dogs are typically larger and more muscular than cats. They have a more varied body shape with different breeds. Cats, on the other hand, are smaller and streamlined, with requir
```

