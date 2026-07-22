# AgentDojo Integration

This folder contains the Stage 6.1 compatibility layer for routing AgentDojo
benchmark tasks through AegisVault runtime security.

The AgentDojo adapter preserves the production runtime security path:

1. Layer 0 request sanity validation
2. Goal Vault initialization
3. Qwen or AgentDojo agent step
4. Layer 0 tool-call validation
5. Sentinel evaluation
6. Action Gate authorization
7. Tool execution
8. AgentDojo evaluator

The semantic Request Gate and Response Gate are not used for AgentDojo runtime
execution. AgentDojo focuses on tool execution security, so the protected path is
Layer 0, Goal Vault, Sentinel, Action Gate, and then tool execution.

Run the mock compatibility smoke test:

```bash
.venv/bin/python evaluation/agentdojo/smoke_agentdojo_adapter.py
```

Run the protected AgentDojo benchmark sample:

```bash
TOKENIZERS_PARALLELISM=false LOCAL_LLM_PORT=11434 .venv/bin/python evaluation/agentdojo/run_pilot_benchmark.py \
  --clean-limit 97 \
  --attack-limit 60 \
  --balanced-by-suite \
  --selection-strategy first \
  --case-layout interleave-types \
  --phase protected \
  --action-timeout-seconds 120 \
  --output-dir evaluation/agentdojo/results/agentdojo_stageA_20clean_10attack_fix3
```

The runner supports resume by default. Reusing the same output directory skips
completed case IDs and continues the remaining cases.

## Latest Protected Run

Latest saved result folder:

```text
evaluation/agentdojo/results/agentdojo_stageA_20clean_10attack_fix3
```

Coverage:

```text
157 total cases
97 clean AgentDojo user tasks
60 injected attack cases
```

Metrics:

```text
Clean utility success: 39.18%
Attack Success Rate / IMP / Tool ASR: 0.00%
Middleware false positives: 5 / 97 = 5.15%
Model utility failures: 54 / 97 = 55.67%
False negatives: 0
```

Per-suite utility on the 157-case protected run:

```text
Workspace: 52.73%
Slack:     61.11%
Travel:    45.71%
Banking:   22.58%
```

## Metric Separation

Do not treat every failed clean AgentDojo task as an AegisVault false positive.

Middleware false positives count only cases where AegisVault incorrectly blocks
or modifies a legitimate clean action, such as a Sentinel or Action Gate block on
a goal-aligned tool call.

Model utility failures are separate. They include wrong dates, wrong tool
arguments, incomplete retrieval, arithmetic mistakes, duplicate tool calls, and
AgentDojo exact-match failures after AegisVault allowed execution.

This distinction matters because AegisVault is the security middleware around
the agent. It does not guarantee that the underlying local Qwen model completes
every AgentDojo clean task correctly.

Manual AgentDojo setup is required before running real Workspace, Slack,
Banking, and Travel suites.
