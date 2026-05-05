# OARS Phase I and Self-Evolve Integration Plan

## Goal
Produce real measurable gains on miniF2F and PutnamBench with a trainable prover stack, not proxy-only metrics.

## Papers Integrated
1. Evolution Strategies as a Scalable Alternative to RL (arXiv:1703.03864)
2. Evolution Strategies at Scale: LLM Fine-Tuning Beyond RL (arXiv:2509.24372)
3. Evolution Strategies at the Hyperscale (arXiv:2511.16652, EGGROLL)
4. SELF: Self-Evolution with Language Feedback (arXiv:2310.00533)
5. A Survey on Self-Evolution of LLMs (arXiv:2404.14387)

## Architecture Changes
1. Generator:
- Supervised fine-tuning of a causal LM on formal statement -> proof traces.
- Add iterative self-refine loop for failed proofs (SELF-style).

2. Evaluator:
- Lean CLI formal verification as the only success criterion for pass@k.
- Compact failure logging with first failing case to avoid truncated runs.

3. Optimizer:
- RL/GRPO baseline for allocator/tactic-policy.
- ES branch for long-horizon sparse rewards.
- EGGROLL-style low-rank perturbations for scalable ES in M3/M4.

## Milestone Mapping
1. M1:
- Train SFT prover and measure pass@k on miniF2F.

2. M2:
- Enable ARLA hardcoded concept bank and compare delta pass-rate vs M1.

3. M3:
- Swap hardcoded bank with discovered concepts.
- Use ES perturbation branch when RL plateaus.

4. M4:
- Train on miniF2F concepts, evaluate transfer on held-out PutnamBench families.

## Acceptance Gates
1. M2 gate:
- Delta pass-rate > 0 and CI95 excludes 0.

2. M3 gate:
- Delta pass-rate discovered > hardcoded and stable across 3 seeds minimum.

3. M4 gate:
- Positive transfer on held-out Putnam family split with formal Lean verification.
