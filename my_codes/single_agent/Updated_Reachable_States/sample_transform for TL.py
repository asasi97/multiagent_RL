## For a 10x10 environment
# Modify it to allow for any size environment
state = [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0] # 969

x = state[0:4]
y = state[4:8]
print('Old X - ',x)
print('Old Y - ',y)

x.reverse()
y.reverse()
print('X - ',x)
print('Y - ',y)


x = int(''.join(map(str, x)),2)
y = int(''.join(map(str, y)),2)
print('Value of X - ',x)
print('Value of Y - ',y)

current_state =(x,y)


## Conversion from Integer to binary
# Have to convert the current state to binary and check the possible actions from TL

x_bin = "{0:b}".format(x).zfill(4) # or format in any length
y_bin = "{0:b}".format(y).zfill(4)

print('X - ',x_bin)
print('Y - ',y_bin)


### Don't really have to split

all_actions_start = canReach.extend(data['nodes']["0"]['trans'])

# loop
for pos_act in all_actions_start:
	pos_act = canReach.extend(data['nodes'][str(node)]['state'])
	l = len(pos_act)
	pos_act = pos_act[l-4:l]
	
allowable_actions = 








