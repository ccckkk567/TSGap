import re
from typing import List

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """Parse ordinary responses without changing the recorded model output."""
    parsed, valids, available = [], [], []
    for response, pool in zip(actions, action_pools):
        matches = re.findall(r"<action>(.*?)</action>", response, re.DOTALL)
        action = matches[0].strip().lower() if matches else "invalid action"
        valid = re.fullmatch(
            r"\s*<think>.+?</think>\s*<action>[^<>]+</action>\s*",
            response, re.DOTALL,
        ) is not None and len(matches) == 1
        parsed.append(action)
        valids.append(valid)
        available.append(action in pool)
    return parsed, valids, [], available


def alfworld_projection_rlvmr(actions: List[str], action_pools: List[List[str]]):
    skill_tags = [
        r"<planning>.*?</planning>",
        r"<reflection>.*?</reflection>",
        r"<explore>.*?</explore>",
        r"<monitor>.*?</monitor>"
    ]

    actions_out = []
    valids = []
    plannings = []
    action_available = [False] * len(actions)
    for i, output in enumerate(actions):
        valid = 1
        act_str = ""

        planning_content = None
        planning_match = re.search(r"<planning>([\s\S]*?)</planning>", output, re.IGNORECASE)
        if planning_match:
            planning_inner = planning_match.group(1).strip()
            planning_content = planning_inner if planning_inner else None
        plannings.append(planning_content)

        # Check for Chinese
        if re.search(r'[\u4e00-\u9fff]', output):
            valid = 0

        # Check ONLY ONE <action>...</action>
        matches = re.findall(r"<action>([\s\S]*?)</action>", output, re.IGNORECASE)
        if len(matches) != 1:
            valid = 0
        else:
            act_candidate = matches[0].strip()
            act_str = act_candidate
            if act_candidate in action_pools[i]:
                action_available[i] = True

        # Check ONLY ONE skill tag, appears before <action> and is non-empty
        found_skill = False
        min_action_pos = output.lower().find("<action>")
        skill_positions = []
        skill_count = 0
        for tag in skill_tags:
            tag_matchs = list(re.finditer(tag, output, re.IGNORECASE | re.DOTALL))
            skill_count += len(tag_matchs)
            for tag_match in tag_matchs:
                if tag_match:
                    # Remove the xml tags and check if not empty
                    inner = re.sub(r"<.*?>", "", tag_match.group(0)).strip()
                    if inner:
                        found_skill = True
                        skill_positions.append(output.find(tag_match.group(0)))
                        # skill tag must appear before <action>
                        if output.find(tag_match.group(0)) > min_action_pos:
                            valid = 0

        if skill_count != 1:  
            valid = 0
        if not found_skill:
            valid = 0

        actions_out.append(act_str)
        valids.append(valid)

    return actions_out, valids, plannings, action_available