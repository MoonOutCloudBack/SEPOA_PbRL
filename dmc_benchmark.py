DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]


TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS

PRIMAL_TASKS = {
    'cheetah': 'cheetah_run',
    'walker': 'walker_run',  # original walker_stand
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'hopper': 'hopper_hop',
    'humanoid': 'humanoid_stand',
    # meta world
    'door-open': 'metaworld_door-open-v2',
    'button-press': 'metaworld_button-press-v2',
    'sweep-into': 'metaworld_sweep-into-v2',
    'drawer-open': 'metaworld_drawer-open-v2',
    'window-open': 'metaworld_window-open-v2',
}


def get_domain(task):
    domain, _ = task.split('_', 1)
    if domain == "metaworld":
        domain = list(PRIMAL_TASKS.keys())[list(PRIMAL_TASKS.values()).index(task)]
    return domain
