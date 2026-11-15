"""Approved English prompt for feedback-driven teaching trajectory refinement."""

TEACHER_REFLECTION_TEMPLATE = """You are a Teacher generating teaching trajectories for a fixed Student model.

Your current teaching trajectory has successfully completed a source task in the ALFWorld environment. The Student used this complete trajectory as an in-context learning (ICL) demonstration to attempt four distinct task instances of the same ALFWorld task type. Each Student attempt started from its own target task's initial environment state. Below are the source demonstration, the Student's actual execution results on the target tasks, and the existing records of teaching iterations.

Your objective:
Use the Student's actual behavior to determine which parts of the current demonstration are worth preserving and which may need adjustment. Provide guidance for generating the next teaching trajectory through a new execution of the source task.

You are not directly correcting an unfinished Student attempt, nor are you merely seeking a shorter or more elegant solution for yourself. The current feedback measures the Student's task performance while using the demonstration, not learning gains after parameter training.

[How the Student actually uses the demonstration]

The Teacher demonstration describes the source task, not the Student's target task. Judge each rollout against its actual target task and observations. Do not assume object locations, object identifiers, or actions completed in the Teacher's source episode are facts about the Student's target episode. Different valid solution paths are not errors.

At every step, the Student sees:
- The complete current Teacher demonstration.
- The current task, current observation, and admissible actions.
- The observation-action records from at most the two most recent turns of its own current execution.

The Student does not see:
- This Teacher reflection.
- Other versions of the teaching trajectory or previous Student rollouts.
- Its own execution history beyond the two most recent turns, or its reasoning outputs from previous steps.

The object you may change is the Teacher's next teaching trajectory. Do not propose changes to the Student model, the common interaction template, the history policy, or the execution budget.

[Current task and execution settings]
{task_and_execution_settings}

[Current teaching demonstration: version {trajectory_version}]
{current_demonstration_text}

[Student results for this round]
{rollout_results}

[Complete Student execution records for this round]
{student_rollout_records}

[Records of previous teaching iterations]
{previous_round_records}

[Reflection requirements]

Base your assessment on both the demonstration and the actual execution records, not only on the aggregate success rate.

Focus on important behaviors supported by the records, and cite rollout identifiers and relevant steps. Compare useful differences between unsuccessful and successful attempts. If either kind of attempt is absent, do not invent a comparison. A path that differs from the Teacher's path is not, by itself, an error.

For each issue you consider important, check whether the current demonstration already provides the relevant information. Distinguish directly observed behavior from a possible explanation involving a limitation of the teaching material. Do not conclude from a small number of results that the Student lacks a particular capability, and do not assume that every failure was caused by the current demonstration.

Make recommendations based on what the Student could actually see at the relevant step. You can inspect the complete logs, but you must not assume the Student could also see the complete logs at that time. Do not interpret system execution errors as evidence of a Student capability problem.

The next version may change explanations, the organization of steps, or a valid execution path. Adding, removing, simplifying, or reorganizing content is allowed, but changes should relate to the observed issues. Do not assume that more length, more detail, or more checks are necessarily more effective. Do not force changes to parts that already work merely to make the trajectory different.

Use previous iterations to identify changes worth retaining and changes that should be reconsidered. Judgments in earlier reflections remain hypotheses; they do not become facts simply because they were written down.

If the current results do not provide a clear basis for improvement, say so directly. If all attempts succeeded, do not invent failures or claim that generalization has been established. If all attempts failed, do not immediately conclude that the Student cannot complete the task. You may recommend preserving the existing approach or propose a tentative adjustment whose usefulness remains to be examined through subsequent results.

Next, the Teacher will restart the source task from its initial environment state and execute it again in the real environment. Your guidance must be realizable through the actions and explanations in that execution. Do not fabricate observations, require deliberate failures to demonstrate recovery, or pass the outer-loop feedback or this reflection directly to the Student as additional hints.

The new demonstration must be understandable on its own, without depending on statements such as "based on the Student's earlier mistake." Do not present facts from old logs as though they have already been observed in the new execution. Any state that needs confirmation should be assessed using real information from the new execution.

[Output format]

## Feedback Analysis
Briefly describe the key execution evidence, possible teaching limitations in the current demonstration, and useful content to retain. State where the evidence is insufficient. Do not repeat the entire log.

## Guidance for the Next Trajectory
Provide specific guidance for the next real execution: what to retain or change, when to express or apply it, and what Student execution behavior it is expected to help improve. Propose only necessary adjustments; there is no required number of recommendations.

Output only the reflection and execution guidance here. Do not generate a new complete environment trajectory in this response.
"""
