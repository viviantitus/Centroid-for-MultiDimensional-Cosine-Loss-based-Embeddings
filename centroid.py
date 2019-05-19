import numpy as np
import math
import collections

early_stop_var = collections.deque(maxlen=3)

a = np.array([[4,3],[5,2],[2,5],[2,2]])
b = np.array([2,2])

for i in range(20):
    
  avg_loss = 0
  for v in range(a.shape[0]):
    loss = a[v].dot(b)/(np.linalg.norm(a[v]) * np.linalg.norm(b))
    avg_loss = avg_loss + loss
    
    
  avg_grads = np.zeros(b.size)
  for v in range(a.shape[0]):
    derivative_dim = []
    for j in range(a.shape[1]):
      tmp1 = a[v][j]/(np.linalg.norm(a[v])*np.linalg.norm(b))
      tmp2 = (a[v].dot(b)*2*b[j])/(np.linalg.norm(a[v]) * pow(np.linalg.norm(b),3))
      derivative_dim.append(tmp1 + tmp2)
    
    for z in derivative_dim:
      avg_grads = avg_grads + z
  grad = np.array(avg_grads/a.shape[0])
  
  
  b = b+(grad)
  early_stop_var.append(avg_loss/a.shape[0])
  
  if math.fabs(avg_loss/a.shape[0] - np.average(early_stop_var)) < 0.01 and len(early_stop_var) == 3:
    print("Last Avg Loss: " + str(avg_loss/a.shape[0]))
    print("Last b: ",b)
    break
  
  if(i%2 == 0):
    print(np.average(early_stop_var))
    print("Avg Loss: " + str(avg_loss/a.shape[0]))
    print("b: ",b)





