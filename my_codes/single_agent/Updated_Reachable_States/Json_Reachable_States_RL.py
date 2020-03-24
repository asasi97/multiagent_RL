import json

with open('json_safeRL_single_agent_one_obstacle_switchedxy') as json_file:
    data = json.load(json_file)

    for node in data['nodes']:
        y = node
        if y == "0":
            canReach = [0]
            canReach.extend(data['nodes'][node]['trans'])
            print(canReach)
            print("init state + reachables = " + str(canReach))

    canReachLast = 0
    numberOfPossibleStates = len(canReach)
    loopCount = 0

    # loop breaks when you encounter the states that you've already encountered
    while len(canReach) != canReachLast: ## Jump from starting node to all the nodes to find the number of reachable states
        loopCount += 1
        canReachLast = len(canReach)
        for node in data['nodes']:
            for state in canReach:
                specificState = state
                if node == str(specificState):
                    # print("yay")
                    for trans in (data['nodes'][node]['trans']):
                        # print(trans)
                        if trans not in canReach:
                            canReach.append(trans)

print("loopCount: " + str(loopCount))
print(canReach)
canReach.sort()
#print("numberOfPossibleStates: " + str(len(canReach)))
#print("All reachable states = " + str(canReach))

start = data['nodes']['0']['trans']
print(start)
print(data['nodes'][str(start[2])]['trans'])


all_actions_start = data['nodes']["0"]['trans']
print(all_actions_start)


tl_actions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
allowable_actions = []
state_allowable_actions = []
# loop
for node in all_actions_start:
    possible_action = data['nodes'][str(node)]['state']
    l = len(possible_action)
    possible_action = possible_action[l-4:l]

    # in except you can add 'NO ACTION' later on if there is no action to be taken
    try:
        index = possible_action.index(1)
        allowable_actions.append(tl_actions[index])
        state_allowable_actions.append(node)
    except ValueError:
        pass   
   
print(allowable_actions)
print(state_allowable_actions)

chose_action = 'RIGHT'
next_node = state_allowable_actions[allowable_actions.index(chose_action)]
print(next_node)

