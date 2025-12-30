import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

points={"blue":[[2,4,3],[1,3,5],[2,3,1],[3,2,3],[2,1,6]],
       "red":[[5,6,5],[4,5,2],[4,6,1],[6,6,1],[5,4,6],[10,10,4]]}


new_point=[5,5,2]

def euclidean_distance(p,q):
    return np.sqrt(np.sum((np.array(p)-np.array(q))**2))

class KNesrestNeighbors:
    def __init__(self,k=3):
        self.k=k
        self.point=None

    def fit(self,point):
        self.point=point
    def predict(self,new_point):
        distances=[]
        for category in self.point:
            for point in self.point[category]:
                distance=euclidean_distance(point,new_point)
                distances.append((distance,category))

        categories=[category[1] for category in sorted(distances)[:self.k]] 
        most_common=Counter(categories).most_common(1)[0][0]

        return most_common

clf=KNesrestNeighbors(k=3)
clf.fit(points)
prediction=clf.predict(new_point)
print("The predicted color is:",prediction)




# Visualization
fig=plt.figure(figsize=(15,12))
ax=fig.add_subplot(projection='3d')
ax.grid(True,color='lightgray')
ax.set_facecolor("#000000")
ax.figure.set_facecolor('white')
ax.tick_params(axis='x',color='gray')
ax.tick_params(axis='y',color='gray')   


for point in points["blue"]:
    plt.scatter(point[0],point[1],point[2],color='blue')

for point in points["red"]:
    plt.scatter(point[0],point[1],point[2],color='red')

new_class=clf.predict(new_point)
color='blue' if new_class=='blue' else 'red'
plt.scatter(new_point[0],new_point[1],new_point[2],color=color,marker='x')

for point in points["blue"]:
    ax.plot([new_point[0],point[0]],[new_point[1],point[1]],[new_point[2],point[2]],color='blue',linestyle='dotted' )
for point in points["red"]:
    ax.plot([new_point[0],point[0]],[new_point[1],point[1]],[new_point[2],point[2]],color='red',linestyle='dotted' )

plt.show()