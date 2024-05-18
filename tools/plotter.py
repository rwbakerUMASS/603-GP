import matplotlib.pyplot as plt
import numpy as np
import pickle

angle = pickle.load(open('angle_metric','rb'))
dist = pickle.load(open('dist_metric','rb'))
human_visible = pickle.load(open('human_visible_metric','rb'))
lidar = pickle.load(open('lidar_metric','rb'))
pose = pickle.load(open('pose_metric','rb'))
metric_map = plt.imread('/home/rwbaker/Downloads/map.png')

people = {}

for i, time in enumerate(lidar):
    x, y = pose[i][0][:2]
    theta = pose[i][1][3]
    for person in time:
        if person.object_id not in people.keys():
            people[person.object_id] = {
                'x':[],
                'y':[],
                't':[]
            }
        mag = np.sqrt(x**2 + y**2)
        people[person.object_id]['x'].append(x + np.cos(theta))
        people[person.object_id]['y'].append(y + np.sin(theta))
        people[person.object_id]['t'].append(i)
for key in people.keys():
    x = people[key]['x']
    y = people[key]['y']
    t = people[key]['t']
    # plt.imshow(metric_map, cmap='gray')
    # plt.scatter(x,y)
    # plt.plot(t,np.arctan2(y,x))
    # plt.plot(angle)
# plt.show()

x = []
y = []
for pos in pose:
    x.append(pos[0][0])
    y.append(pos[0][1])
    # plt.imshow(metric_map, cmap='gray')
# plt.scatter(x,y)
# plt.show()

N=10
plt.plot(np.convolve(np.array(angle)[100:300],np.ones(N)/N))
plt.hlines([0],xmin=0,xmax=200,colors=['black'])
plt.ylabel('Angle (rad)')
plt.xlabel('Time (s)')
plt.savefig('/home/rwbaker/catkin_ws/src/603-GP/res/angle.png')
plt.show()

dist = np.array(dist)[100:300]
for i in range(200):
    if dist[i] is None:
        dist[i] = dist[i-1]

plt.plot(np.convolve(dist+200,np.ones(N)/N))
plt.ylabel('Dist (mm)')
plt.xlabel('Time (s)')
plt.hlines([1000],xmin=0,xmax=200,colors=['black'])
plt.savefig('/home/rwbaker/catkin_ws/src/603-GP/res/dist.png')
plt.show()

plt.plot(np.convolve(np.array(human_visible)[100:300],np.ones(N)/N,'valid'))
plt.ylabel('Percent of Time Human is Visible')
plt.xlabel('Time (s)')
plt.savefig('/home/rwbaker/catkin_ws/src/603-GP/res/visible.png')
plt.show()