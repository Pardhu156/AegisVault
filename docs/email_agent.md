# Stage 4.1 Email Agent

Stage 4.1 builds the first realistic baseline agent on top of the generic Stage 4.0 runtime. It does not integrate AegisVault guardrails yet.

## Architecture

```text
User task
  -> Qwen through Ollama
  -> Generic AgentRuntime
  -> Email Tool Registry
  -> Synthetic threaded email dataset
  -> Final response and trace
```

The runtime remains generic. Email behavior comes from `aegisvault.email_agent`, the email-specific system prompt, and tools backed by `datasets/email`.

## Dataset

Email data lives under:

```text
datasets/email/
  inbox/emails.jsonl
  sent/sent_emails.jsonl
  drafts/drafts.jsonl
  contacts/contacts.json
  tasks/email_tasks.jsonl
```

The mailbox contains realistic threaded conversations with reply and forward history, including manager, HR, finance, invoices, travel, meetings, Project Alpha, Project Beta, newsletters, system notifications, support, recruitment, and client messages.

## Tools

The Email Agent exposes:

- `search_email`
- `read_email`
- `summarize_email`
- `draft_email`
- `send_email`
- `delete_email`
- `archive_email`
- `list_unread`
- `list_threads`

`send_email` is simulated. It appends to the local synthetic sent dataset and never sends real email.

## Commands

List email tools:

```bash
python run_agent.py --email-agent --list-tools
```

Run one task:

```bash
python run_agent.py --email-agent --task "Summarize unread emails from my manager." --model qwen3:4b-instruct
```

For slower local machines, cap generation and increase timeout:

```bash
python run_agent.py --email-agent --task "Find invoices from Amazon." --model qwen3:4b-instruct --timeout-seconds 180 --num-predict 160
```

Run interactive mode:

```bash
python run_agent.py --email-agent --interactive --model qwen3:4b-instruct
```

Run fast mock evaluation:

```bash
python evaluation/agent_runtime/scripts/run_email_agent_evaluation.py --mock --limit 5
```

Run real Qwen evaluation:

```bash
python evaluation/agent_runtime/scripts/run_email_agent_evaluation.py --model qwen3:4b-instruct
```

Small real-model smoke test:

```bash
python evaluation/agent_runtime/scripts/run_email_agent_evaluation.py --model qwen3:4b-instruct --limit 3 --timeout-seconds 180 --num-predict 160
```

Reports are written under:

```text
reports/email_agent/<run_id>/
  conversation_traces.jsonl
  task_results.jsonl
  metrics.json
  runtime_summary.md
```

## Metrics

The evaluation runner records task success rate, tool selection accuracy, average latency, median latency, min/max latency, tool execution count, per-tool usage, average tools per task, runtime failures, and total runtime.

## Stage Boundary

Stage 4.1 intentionally does not use Request Gate, Response Gate, Goal Vault, Action Gate, Redis, policy YAML, or AegisVault middleware. Those integrations belong to later stages.
