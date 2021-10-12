import math
import numpy as np

softmax_output = [[0.7, 0.1, 0.2],
                  [0.7, 0.1, 0.3],
                  [0.7, 0.1, 0.4]]

final_output = []
for s0 in softmax_output:
    intermediate_output = []
    for s1 in s0:
        s_exp = np.exp(s1)
        other_exp = 0
        all_exp = 0
        for s2 in s0:
            if s1 != s2:
                other_exp += np.exp(s2)
            all_exp += np.exp(s2)
        intermediate_output.append((s_exp * other_exp) / np.square(all_exp))
    final_output.append(intermediate_output)
print(final_output)



# With one hot encoding
# target_output = [1, 0, 0]
#
# loss = -(math.log(softmax_output[0]) * target_output[0] +
#           math.log(softmax_output[1]) * target_output[1] +
#           math.log(softmax_output[2]) * target_output[2])
#
# print(-math.log(0.7))
# print(-math.log(0.5))


