**CAMRL --- Cooperative Multi-Agent Reinforcement Learning Lab**
Project progress documentation
Documentation cut-off: 9 November 2026
Current stopping point: Phase 14.9 complete; Phase 14.10 diagnostics
prepared but not yet run.
1. Project overview
CAMRL (Cooperative Multi-Agent Reinforcement Learning Lab) is an
experimental platform for comparing cooperative multi-agent
reinforcement learning algorithms in a custom Cooperative Resource
Management Environment (CRME).

The algorithms planned for comparison are:
- Independent Proximal Policy Optimization (IPPO)
- Multi-Agent Proximal Policy Optimization (MAPPO), using centralized
  training and decentralized execution (CTDE)
- QMIX, using monotonic value decomposition
- Communication-enabled MAPPO (Comm-MAPPO), with learned inter-agent
  communication and centralized training
The current work has focused on validating the environment, implementing
and auditing the training pipelines, isolating baseline and experimental
reward-shaping branches, and running an initial QMIX comparison.
2. Development environment and workflow
- Primary development environment: Google Colab Pro
- GPU reported during project setup: NVIDIA L4, approximately 22.5 GB
  GPU memory
- Framework: PyTorch
- Checkpoints and experiment outputs: Google Drive
- Preferred workflow: phased notebook development, with validation
  output checked before proceeding to longer training runs
The intended future application platform is:
- Frontend: Next.js and TypeScript
- Backend: FastAPI
- Database: PostgreSQL / Cloud SQL
- Model checkpoints: Google Drive / Google Cloud Storage
- Deployment: Docker and Cloud Run
- CI/CD: GitHub Actions
- Experiment tracking: MLflow or Weights & Biases
These are planned platform components, not all implemented as part of
the current experiment.
3. Cooperative Resource Management Environment (CRME) V2
3.1 Environment configuration
  Property                        Configuration
  Grid                                  12 × 12
Agents                                      4
Resources                                  10
Obstacles                                  12
Maximum episode length              200 steps
Initial agent energy                      100
Actions per agent                           6
Local observation dimension               195
Global state dimension                    780
3.2 Action space
Action Meaning

     0 Stay
     1 Move up
     2 Move down
     3 Move left
     4 Move right
     5 Collect
A normal resource is collected when one agent occupies the exact
resource cell and chooses the collect action.

A special resource requires two agents to occupy the exact same resource
cell and both choose collect.
3.3 Observation structure
Each agent receives a 195-dimensional local observation:

  Index range   Contents
  0--2          Own x position, y position, and energy
3--42         Ten resources × four features: x, y, collected, special
43--50        Positions of all four agents, two coordinates each
51--194       Flattened 12 × 12 obstacle map

The global state is formed by concatenating the four local observations:

GLOBAL_STATE_DIM = 4 × 195 = 780
3.4 Environment mechanics
The V2 step function returns the five-value API:

(observations, rewards, terminated, truncated, info)

Current reward and energy mechanics:

  Event                               Value
  Valid movement penalty              -0.01
Stay penalty                        -0.02
Invalid movement penalty            -0.05
Movement energy cost                  0.2
Collection energy cost                0.5
Unsuccessful collection penalty     -0.05
Normal resource reward                  2
Special resource reward                 5
All-resources-completed bonus          10

Resource rewards are shared across agents. The episode terminates when
all resources are collected and truncates at the maximum step count.
3.5 Team potential and reward shaping
The team potential is based on Manhattan distance:
- For a normal resource, use the distance to the nearest active agent.
- For a special resource, use the sum of distances from the two
  nearest active agents.
The original shaping formula is:

F(s, s') = 0.1 × (0.99 × Φ(s') − Φ(s))

An experimental alternative was implemented:

F_exp(s, s') = 0.1 × (Φ(s') − Φ(s))

The experimental formula is treated as an ablation for empirical
comparison. The original discounted formula is the conventional
potential-based shaping form. No claim has been established that the
alternative is theoretically superior.
3.6 Environment validation
The V2 environment passed controlled checks for normal and special
resource collection. The observation dimensions, action mechanics, and
environment transitions were validated.
4. Model dimensions and algorithm implementations
Shared dimensions:
OBS_DIM = 195
NUM_AGENTS = 4
ACTION_DIM = 6
GLOBAL_STATE_DIM = 780
MESSAGE_DIM = 64
The implementations include:
- IPPO: four independent Actor-Critic agents.
- MAPPO: four decentralized actors and one centralized critic
  operating on the global state.
- Comm-MAPPO: four communication actors, message dimension 64, and a
  centralized critic.
- QMIX: four agent Q-networks, a mixing network, target agent
  networks, and a target mixing network.
The QMIX mixing network was tested for monotonicity.
5. Replay buffers, rollout collection, and update validation
5.1 Episode replay buffer
The QMIX replay buffer is episode-based and exposes:
- add_episode(episode)
- sample(batch_size)
- len(replay_buffer)
The buffer stores episodes and samples transitions from randomly
selected episodes.
5.2 Training update validation
The following update paths were smoke-tested successfully:
- IPPO optimizer update
- MAPPO optimizer update
- Comm-MAPPO optimizer update
- QMIX V2 optimizer update
Comm-MAPPO's differentiable joint logits and communication messages were
corrected. A gradient audit confirmed gradients reached all four
communication actors.

QMIX target calculation was corrected to reshape reward and termination
tensors to (-1, 1), resolving a broadcasting issue. A QMIX update
passed with a reported loss of approximately 0.0056401 during the
earlier smoke test.
5.3 Termination and truncation handling
The PPO-style collectors distinguish true termination from time-limit
truncation:
- True termination does not bootstrap the next value.
- Truncation bootstraps the next value.
- Generalized Advantage Estimation stops across episode boundaries.
5.4 Persistent rollout issue and correction
The original IPPO, MAPPO, and Comm-MAPPO collectors reset their
environments at the beginning of every collection call. This could break
episodes across rollout boundaries.

Persistent collectors were introduced to preserve observations and
running episode returns between calls. The continuity test passed:

  Algorithm      First rollout   Second rollout   Persistent total
  IPPO                20 steps         40 steps                 40
MAPPO               20 steps         40 steps                 40
Comm-MAPPO          20 steps         40 steps                 40

This verified that rollout calls continue the same environment
trajectory rather than resetting at every call.
6. Baselines and diagnostic findings before reward-shaping comparison
6.1 Initial benchmark
The initial V1 benchmark showed poor learned-policy performance:
- IPPO, MAPPO, and Comm-MAPPO: zero reward and zero success in the
  reported benchmark.
- QMIX: mean reward approximately -13.3997.
- Random baseline: mean reward approximately +3.2937.
The V1 environment and training pipeline were subsequently revised.
6.2 Heuristic benchmark
The observation-only centralized planner and oracle heuristic each
achieved:
- 20 successes in 20 seeds
- 100% completion
- Mean episode length: 49.6 steps
- Episode length range: 41--59 steps
- Mean episode return: approximately 49.743
The observation-only planner reads all four agents' observations
centrally. It is not a decentralized learned policy.

This result indicates that the environment can be solved by a
coordinated planner and that the main challenge is currently in learned
policy training and coordination.
6.3 Learned policy diagnostics
The earlier learned policies showed zero resource collection during
evaluation. Action audits indicated that the policies generally did not
select collect when positioned on resource cells.

A trajectory audit found:
- IPPO: no valid collection attempts.
- MAPPO: no valid collection attempts.
- Comm-MAPPO: no valid collection attempts.
- QMIX: no valid collection attempts.
The policies were moving, but were not successfully executing collection
behavior.
7. Checkpoints and experiment artifacts
The following files were saved in Google Drive:
7.1 Original baseline checkpoint
/content/drive/MyDrive/CAMRL/checkpoints/CAMRL_baseline_20261109_113417.pt

Reported size: approximately 45.55 MB.

It contains model weights and optimizer states for IPPO, MAPPO,
Comm-MAPPO, and QMIX.
7.2 Training pilot checkpoint
/content/drive/MyDrive/CAMRL/checkpoints/CAMRL_training_pilot_20261109_114443.pt
7.3 Training pilot evaluation CSV
/content/drive/MyDrive/CAMRL/checkpoints/CAMRL_training_pilot_results_20261109_114443.csv
7.4 QMIX comparison pilot CSV
/content/drive/MyDrive/CAMRL/checkpoints/CAMRL_QMIX_comparison_20261109_115859.csv
7.5 QMIX branch checkpoint
/content/drive/MyDrive/CAMRL/checkpoints/CAMRL_QMIX_branches_20261109_115859.pt
7.6 QMIX deterministic evaluation CSV
/content/drive/MyDrive/CAMRL/checkpoints/CAMRL_QMIX_evaluation_20261109_120002.csv

These are paths reported by the Colab session. The current documentation
records them; it does not independently verify their continued
availability.
8. Baseline and experimental branch isolation
The original baseline checkpoint was restored and cloned into two
independent branches:
- baseline_branch
- experimental_branch
The branches contain independent model and optimizer objects. Initial
model weights were verified to be identical.

The branch activation helper switches the active global model and
optimizer references used by the existing training functions.

Separate QMIX replay buffers were created:
- baseline_qmix_replay
- experimental_qmix_replay
The replay isolation test passed: adding a temporary episode to the
baseline replay did not change the experimental replay.
8.1 Paired environment validation
Two separate environments were created:
- Baseline environment with the original step function.
- Experimental environment with the alternative reward-shaping
  wrapper.
With identical initial seeds and identical stay actions, validation
passed for:
- Identical initial observations.
- Identical environment transitions.
- Matching resource and termination state.
- Different reward shaping formulas.
For the five-step stay test:

  Metric             Baseline   Experimental
  Mean reward           +0.04          -0.02
Shaping reward        +0.06           0.00

The baseline retained positive discounted potential shaping despite no
change in potential. The experimental difference-based shaping was zero
when potential did not change.

A validation cell initially raised a KeyError because it expected
team_potential_before and team_potential_after in the baseline
info dictionary. Those fields are only added by the experimental
wrapper. The underlying environment tests had already passed; the issue
was in the validation code.
9. QMIX comparative training pilot --- Phase 14.8A
A short QMIX pilot was run with 20 episodes per branch.
9.1 Configuration
  Setting                                              Value
  Episodes per branch                                     20
Maximum steps per episode                              200
Initial epsilon                                       0.30
Minimum epsilon                                       0.05
Epsilon decay                                         0.97
Updates per episode after replay threshold               2
Replay threshold                               10 episodes
Batch size                                             128
Reward scale in QMIX update                            0.2

The pilot trained the baseline branch first, followed by the
experimental branch. Each branch used its own model objects, optimizer,
and replay buffer.
9.2 Training results
  Metric                         Baseline QMIX   Experimental QMIX
  Episodes                                  20                  20
Mean episode return                  +8.9368             -5.6839
Total resources collected                 16                  20
Mean resources per episode              0.80                1.00
Successful episodes                        0                   0
Final reported update loss          0.001364            0.001993

The baseline branch's losses were reported from episode 10 onward, after
the replay threshold was reached. The experimental branch also began
updating at episode 10.
9.3 Interpretation
The experimental branch collected more resources during exploratory
training (20 versus 16). The baseline branch had a higher mean raw
return.

These results do not establish that either reward design is better:
- The reward functions have different scales and incentives.
- The training run was short.
- No episode completed successfully.
- Exploration may have contributed to resource collection.
- The branches were trained sequentially.
- The pilot did not explicitly pair every training episode with the
  same seed across branches.
The loss values confirm that updates ran, but low loss is not evidence
of a better policy by itself.
10. QMIX deterministic evaluation --- Phase 14.9
Both trained QMIX branches were evaluated using greedy action selection
(epsilon=0) over 20 evaluation seeds.
10.1 Evaluation results
  Metric                        Baseline QMIX   Experimental QMIX
  Mean return                       +5.543675           -9.322500
Mean resources collected                0.0                 0.0
Total resources collected                 0                   0
Success rate                             0%                  0%
Mean episode length                     200                 200

Evaluation seeds began at 20261200 and increased by one for each
episode.
10.2 Interpretation
Neither trained QMIX policy collected resources during deterministic
evaluation, despite collecting some resources during epsilon-greedy
training.

This indicates that the current greedy policies have not learned a
reliable collection strategy.

Potential explanations to investigate include:
- The greedy action selection does not choose collect at the correct
  time.
- The agents do not reliably navigate to resource cells.
- The policies do not coordinate special-resource collection.
- The short pilot did not provide enough useful learning experience.
- The current training and exploration setup may not produce a robust
  greedy policy.
The deterministic evaluation result is more relevant to learned policy
deployment than the exploratory training collection count.
11. Current project status
Completed and validated
- [x] CRME V2 environment mechanics and observation dimensions
- [x] Normal and special resource collection checks
- [x] IPPO, MAPPO, Comm-MAPPO, and QMIX model setup
- [x] QMIX monotonicity check
- [x] PPO and QMIX optimizer smoke tests
- [x] Comm-MAPPO gradient-flow audit
- [x] Termination and truncation bootstrap corrections
- [x] Persistent IPPO, MAPPO, and Comm-MAPPO rollout collectors
- [x] Persistent rollout continuity validation
- [x] Baseline checkpoint restoration
- [x] Independent baseline and experimental model branches
- [x] Separate QMIX replay buffers
- [x] Branch activation and replay isolation validation
- [x] Paired environment and reward-function validation
- [x] 20-episode baseline QMIX training pilot
- [x] 20-episode experimental QMIX training pilot
- [x] 20-seed deterministic QMIX evaluation
Not yet completed
- [ ] Run Phase 14.10 greedy policy diagnostics
- [ ] Determine whether greedy QMIX policies select the collect action
- [ ] Check whether collect actions occur while agents occupy resource
  cells
- [ ] Investigate special-resource coordination
- [ ] Review QMIX action selection and training implementation if
  diagnostics indicate a defect
- [ ] Design a longer, controlled comparative training experiment
- [ ] Evaluate all four algorithms under a consistent protocol
- [ ] Produce final statistical comparisons and research conclusions
12. Exact next step: Phase 14.10
The next notebook cell has been prepared but has not yet been
executed.

It diagnoses each trained QMIX branch over five seeds and records:
- Overall action distribution.
- Per-agent action distribution.
- Number of collect actions.
- Number of collect actions issued while an agent occupies an active
  resource cell.
- Resources collected.
- Episode returns and lengths.
The planned calls are:
baseline_diagnostics = diagnose_qmix_branch(
    baseline_branch,
    baseline_qmix_replay,
    experimental=False
)

experimental_diagnostics = diagnose_qmix_branch(
    experimental_branch,
    experimental_qmix_replay,
    experimental=True
)
The diagnostic results should guide the next change. Do not assume that
increasing training duration alone will solve the problem before
checking the action-selection and collection behavior.
13. Research reporting cautions
When writing up the experiment:
1. Distinguish exploratory training performance from deterministic
   evaluation performance.
2. Do not compare raw returns across reward functions as if they were
   directly equivalent.
3. Report resource collection and completion rate as separate outcome
   measures.
4. Treat the 20-episode QMIX run as a pilot, not a final benchmark.
5. Use matched seeds and consistent evaluation conditions for the final
   comparison.
6. Report the alternative reward function as an empirical ablation, not
   as a theoretically superior replacement for standard potential-based
   shaping.
7. Do not claim successful learned coordination until policies
   demonstrate repeatable resource collection and episode completion.
End of current progress record --- 9 November 2026.

The work is intentionally paused after Phase 14.9. The next action is to
run the prepared Phase 14.10 QMIX greedy-policy diagnostics.
