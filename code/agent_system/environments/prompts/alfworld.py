# projection_f# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE_NO_HIS_NOTHINK = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE_NOTHINK = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should choose an admissible action for current step and present it within <action> </action> tags.
"""

# ALFWORLD_TEMPLATE_NO_HIS_MC = """
# You are an expert agent operating in the ALFRED Embodied Environment.
# Your current observation is: {current_observation}
# Your admissible actions of the current situation are: [{admissible_actions}].

# Now it's your turn to take an action, following these steps:

# 1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

# <planning>
# Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
# Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
# </planning>

# <explore>
# When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
# Use this approach when facing obstacles that require creative and innovative thinking.
# </explore>

# <reflection>
# Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'Nothing happens' indicates the action is invalid.
# This is typically used when several consecutive actions yield no substantial progress. 
# </reflection>

# <proceed>
# Proceed to the next step based on the prior overall plan or the most recent unfinished exploration.
# This is most often used when the model clearly knows what to do next.
# </proceed>

# 2. After your reasoning, you MUST select and present an admissible action for the current step within <action> </action> tags.

# Specify the next action the agent should take to progress toward the task goal, following these guidelines:
# 1. Object and Receptacle References: Use specific identifiers:
# - [obj id] for objects (e.g., apple 1).
# - [recep id] for receptacles (e.g., countertop 1).
# 2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:
# Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

# <action>
# Choose the most appropriate action from the valid actions.
# </action>
# """

ALFWORLD_TEMPLATE_NO_HIS_MC = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

<planning>
Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
It is necessary to list all the points separately. eg, step 1: xxx, step 2: xxx, step 3: xxx, etc.
</planning>

<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
Use this approach when facing obstacles that require creative and innovative thinking.
</explore>

<reflection>
Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'No known action matches that input.' indicates the action is invalid.
This is typically used when several consecutive actions yield no substantial progress. 
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task. Recall the current subgoal and consider the next concrete action, ensuring agent alignment with the overall plan.  
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

2. After your reasoning, you MUST select and present an admissible action for the current step within <action> </action> tags.

Specify the next action the agent should take to progress toward the task goal, following these guidelines:
1. Object and Receptacle References: Use specific identifiers:
- [obj id] for objects (e.g., apple 1).
- [recep id] for receptacles (e.g., countertop 1).
2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:
Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

<action>
Choose the most appropriate action from the valid actions.
</action>
"""

ALFWORLD_TEMPLATE_MC = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Your previous overall plan is: {planning}. Please strictly adhere to your plan.

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

<planning>
Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
It is necessary to list all the points separately. eg, step 1: xxx, step 2: xxx, step 3: xxx, etc.
</planning>

<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
Use this approach when facing obstacles that require creative and innovative thinking.
</explore>

<reflection>
Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'No known action matches that input.' indicates the action is invalid.
This is typically used when several consecutive actions yield no substantial progress. 
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task. Recall the current subgoal and consider the next concrete action, ensuring agent alignment with the overall plan.  
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

2. After your reasoning, you MUST select and present an admissible action for the current step within <action> </action> tags.

Specify the next action the agent should take to progress toward the task goal, following these guidelines:
1. Object and Receptacle References: Use specific identifiers:
- [obj id] for objects (e.g., apple 1).
- [recep id] for receptacles (e.g., countertop 1).
2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:
Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

<action>
Choose the most appropriate action from the valid actions.
</action>
"""
TEACHING_STEP_TEMPLATE = """You are acting in the ALFWorld household environment.
Complete the task using the actual observations and admissible actions.
At each step output exactly one concise <think>...</think> followed by
one <action>...</action>. Choose an action from the current admissible actions.

{context}
Task: {task}
Step: {step_index}
Recent observation-action history:
{history}

Current observation:
{observation}

Current admissible actions:
{admissible_actions}
"""

TEACHER_OBJECTIVE = """Complete the task in the real environment and produce a
self-contained demonstration that another agent can understand."""


def build_step_prompt(task, observation, admissible_actions, history, step_index,
                      demonstration="", teacher_guidance=""):
    recent = history[-2:]
    rendered_history = []
    for offset, entry in enumerate(recent):
        number = step_index - len(recent) + offset
        previous_observation = entry.get("observation", entry.get("text_obs"))
        rendered_history.append(
            f"Step {number}\nObservation: {previous_observation}\nAction: {entry['action']}"
        )
    context = []
    if demonstration:
        context.append(f"Complete demonstration:\n{demonstration}")
    if teacher_guidance:
        context.append(f"Execution guidance:\n{teacher_guidance}")
    return TEACHING_STEP_TEMPLATE.format(
        context="\n\n".join(context), task=task, step_index=step_index,
        history="\n".join(rendered_history) or "No previous actions.",
        observation=observation, admissible_actions="\n".join(admissible_actions),
    )


def render_demonstration(episode):
    parts = [f"Task: {episode['task']}",
             f"Initial observation: {episode['initial_observation']}"]
    for step in episode["steps"]:
        parts.append(
            f"Step {step['step']}\nObservation: {step['observation']}\n"
            f"Response: {step['response']}\nAction: {step['action']}\n"
            f"Environment returned: {step['next_observation']}"
        )
    return "\n\n".join(parts)
