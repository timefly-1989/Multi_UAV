import numpy as np
import pyglet
import math
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

class Env(object):
    viewer = None
    region_x = 600
    region_y = 600
    region_xy = np.sqrt(region_x ** 2 + region_y ** 2)
    dt = 1  # refresh rate
    charging_acceleration_x_bound = [-1, 1]
    charging_acceleration_y_bound = [-1, 1]
    charging_speed_x_bound = [-2, 2]
    charging_speed_y_bound = [-2, 2]
    goal_acceleration_x_bound = [-0.1, 0.1]
    goal_acceleration_y_bound = [-0.1, 0.1]
    goal_speed_x_bound = [-0.2, 0.2]
    goal_speed_y_bound = [-0.2, 0.2]

    def __init__(self, charging_num, goal_num):
        self.charging_infos = np.zeros(charging_num, dtype=[('ID', np.int), ('speed_x', np.float32), ('speed_y', np.float32), ('position_x', np.float32), ('position_y', np.float32), ('acceleration_x', np.float32), ('acceleration_y', np.float32)])
        self.goal_infos = np.zeros(goal_num, dtype=[('ID', np.int), ('speed_x', np.float32), ('speed_y', np.float32), ('position_x', np.float32), ('position_y', np.float32)])

    def reset(self):
        obslist = []
        id = 1
        for c in self.charging_infos:
            c['speed_x'] = 0.
            c['speed_y'] = 0.
            c['acceleration_x'] = 0.
            c['acceleration_y'] = 0.
            c['position_x'] = np.random.uniform(0, self.region_x)
            c['position_y'] = np.random.uniform(0, self.region_y)
            c['ID'] = id
            id = id+1
            obslist.append(c['position_x'])
            obslist.append(c['position_y'])
            obslist.append(c['speed_x'])
            obslist.append(c['speed_y'])
            obslist.append(c['acceleration_x'])
            obslist.append(c['acceleration_y'])
        id = 1
        for g in self.goal_infos:
            g['speed_x'] = 0.
            g['speed_y'] = 0.
            g['position_x'] = np.random.uniform(0, self.region_x)
            g['position_y'] = np.random.uniform(0, self.region_y)
            g['ID'] = id
            id = id+1
            for c in self.charging_infos:
                obslist.append(c['position_x'] - g['position_x'])
                obslist.append(c['position_y'] - g['position_y'])
                obslist.append(g['speed_x'])
                obslist.append(g['speed_y'])
        return obslist

    def step(self, actions):
        actions = np.array(actions)
        actions = actions.reshape((int(actions.size/2), 2))
        done = True
        reward = 0
        obslist = []
        E_matrix = [[0 for col in range(len(self.charging_infos))] for row in range(len(self.goal_infos))]
        E_matrix_new = [[0 for col in range(len(self.charging_infos))] for row in range(len(self.goal_infos))]
        for i in range(len(actions)):
            self.charging_infos[i]['acceleration_x'] = actions[i][0]
            self.charging_infos[i]['acceleration_y'] = actions[i][1]
            charging_acceleration_x = np.clip(self.charging_infos[i]['acceleration_x'], *self.charging_acceleration_x_bound)
            charging_acceleration_y = np.clip(self.charging_infos[i]['acceleration_y'], *self.charging_acceleration_y_bound)
            self.charging_infos[i]['acceleration_x'] = charging_acceleration_x
            self.charging_infos[i]['acceleration_y'] = charging_acceleration_y
            (charging_speed_x, charging_speed_y) = (self.charging_infos[i]['speed_x'], self.charging_infos[i]['speed_y'])
            (charging_position_x, charging_position_y) = (self.charging_infos[i]['position_x'], self.charging_infos[i]['position_y'])
            charging_speed_x_ = charging_speed_x + self.charging_infos[i]['acceleration_x'] * self.dt
            charging_speed_y_ = charging_speed_y + self.charging_infos[i]['acceleration_y'] * self.dt

            j = 0
            for g in self.goal_infos:
                (goal_position_x, goal_position_y) = (g['position_x'], g['position_y'])
                d = np.sqrt((charging_position_x - goal_position_x) ** 2 + (charging_position_y - goal_position_y) ** 2)
                E_matrix[i][j] = d
                j = j+1

            if charging_speed_x_ > np.max(self.charging_speed_x_bound):
                charging_speed_x_ = np.max(self.charging_speed_x_bound)
            if charging_speed_x_ < np.min(self.charging_speed_x_bound):
                charging_speed_x_ = np.min(self.charging_speed_x_bound)
            if charging_speed_y_ > np.max(self.charging_speed_y_bound):
                charging_speed_y_ = np.max(self.charging_speed_y_bound)
            if charging_speed_y_ < np.min(self.charging_speed_y_bound):
                charging_speed_y_ = np.min(self.charging_speed_y_bound)
            charging_position_x_ = charging_position_x + charging_speed_x * self.dt + self.charging_infos[i]['acceleration_x'] * np.square(self.dt) / 2
            charging_position_y_ = charging_position_y + charging_speed_y * self.dt + self.charging_infos[i]['acceleration_y'] * np.square(self.dt) / 2
            self.charging_infos[i]['speed_x'] = charging_speed_x_
            self.charging_infos[i]['speed_y'] = charging_speed_y_
            self.charging_infos[i]['position_x'] = charging_position_x_
            self.charging_infos[i]['position_y'] = charging_position_y_
            obslist.append(charging_position_x_)
            obslist.append(charging_position_y_)
            obslist.append(charging_speed_x_)
            obslist.append(charging_speed_y_)
            obslist.append(charging_acceleration_x)
            obslist.append(charging_acceleration_y)
        
        E_matrix =np.array(E_matrix)
        row_ind,col_ind=linear_sum_assignment(E_matrix)

        #print(E_matrix[row_ind,col_ind])#提取每个行索引的最优指派列索引所在的元素，形成数组
        #print(E_matrix[row_ind,col_ind].sum())#数组求和
        temp = 0
        j = 0
        for g in self.goal_infos:
            goal_acceleration_x = np.random.uniform(np.min(self.goal_acceleration_x_bound), np.max(self.goal_acceleration_x_bound))
            goal_acceleration_y = np.random.uniform(np.min(self.goal_acceleration_y_bound), np.max(self.goal_acceleration_y_bound))
            goal_acceleration_x = np.clip(goal_acceleration_x, *self.goal_acceleration_x_bound)
            goal_acceleration_y = np.clip(goal_acceleration_y, *self.goal_acceleration_y_bound)
            (goal_speed_x, goal_speed_y) = (g['speed_x'], g['speed_y'])
            (goal_position_x, goal_position_y) = (g['position_x'], g['position_y'])
            goal_speed_x_ = goal_speed_x + goal_acceleration_x * self.dt
            goal_speed_y_ = goal_speed_y + goal_acceleration_y * self.dt
            goal_speed_x_ = np.clip(goal_speed_x_, *self.goal_speed_x_bound)
            goal_speed_y_ = np.clip(goal_speed_y_, *self.goal_speed_y_bound)
            goal_position_x_ = goal_position_x + goal_speed_x * self.dt + goal_acceleration_x * np.square(self.dt) / 2
            goal_position_y_ = goal_position_y + goal_speed_y * self.dt + goal_acceleration_y * np.square(self.dt) / 2
            if goal_position_x_ < 0:
                goal_position_x_ = 0
                goal_speed_x_ = 0
                goal_speed_y_ = 0
            if goal_position_x_ > self.region_x:
                goal_position_x_ = self.region_x
                goal_speed_x_ = 0
                goal_speed_y_ = 0
            if goal_position_y_ < 0:
                goal_position_y_ = 0
                goal_speed_x_ = 0
                goal_speed_y_ = 0
            if goal_position_y_ > self.region_y:
                goal_position_y_ = self.region_y
                goal_speed_x_ = 0
                goal_speed_y_ = 0
            g['speed_x'] = goal_speed_x_
            g['speed_y'] = goal_speed_y_
            g['position_x'] = goal_position_x_
            g['position_y'] = goal_position_y_

            i = 0
            for c in self.charging_infos:
                obslist.append(c['position_x'] - g['position_x'])
                obslist.append(c['position_y'] - g['position_y'])
                obslist.append(g['speed_x'])
                obslist.append(g['speed_y'])
                d = np.sqrt((c['position_x'] - g['position_x']) ** 2 + (c['position_y'] - g['position_y']) ** 2)
                E_matrix_new[i][j] = d
                i = i+1
            j = j+1
        E_matrix_new_sum = 0
        for i in range(len(col_ind)):
            if E_matrix_new[i][col_ind[i]] > E_matrix[i][col_ind[i]]:
                reward = reward - E_matrix_new[i][col_ind[i]] + E_matrix[i][col_ind[i]]
            else:
                E_matrix_new_sum = E_matrix_new_sum + E_matrix_new[i][col_ind[i]]
        if reward < 0:
            pass
        else:
            reward = (E_matrix[row_ind,col_ind].sum() - E_matrix_new_sum)*10

        E_matrix_new =np.array(E_matrix_new)
        row_ind_new,col_ind_new=linear_sum_assignment(E_matrix_new)
        #print(row_ind_new)#开销矩阵对应的行索引
        #print(col_ind_new)#对应行索引的最优指派的列索引
        #print(E_matrix_new[row_ind_new,col_ind_new])#提取每个行索引的最优指派列索引所在的元素，形成数组
        #print(E_matrix_new[row_ind_new,col_ind_new].sum())#数组求和
        #if (E_matrix[row_ind,col_ind].sum() - E_matrix_new[row_ind_new,col_ind_new].sum())>0:
        #    reward = 100
        #else:
        #    reward = -100

        if E_matrix_new[row_ind_new,col_ind_new].sum() < 1:
            done = True
        return obslist, reward, done

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.charging_infos, self.goal_infos, self.region_x, self.region_y)
        self.viewer.render()

    def sample_action(self, charging_num):
        return [[np.random.uniform(-1, 1)] * 2 for i in range(charging_num)]


class Viewer(pyglet.window.Window):
    def __init__(self, charging_infos, charging_goal_infos, x, y):
        self.charging_goal_infos = charging_goal_infos
        self.charging_infos = charging_infos
        pyglet.resource.path = ['resources']
        pyglet.resource.reindex()
        charging = pyglet.resource.image('charging.png')
        user = pyglet.resource.image('user.png')
        radiation = pyglet.resource.image('radiation.png')
        super(Viewer, self).__init__(width=x, height=y, resizable=False, caption='MADDPG', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        
        self.chargings = []
        self.chargings_radiation = []
        self.charging_goals = []
        for u in self.charging_infos:
            charging_sprite = pyglet.sprite.Sprite(img=charging, x=u['position_x'] - 14., y=u['position_y'] - 9.)
            radiation_sprite = pyglet.sprite.Sprite(img=radiation, x=u['position_x'] - 100., y=u['position_y'] - 100.)
            self.chargings.append(charging_sprite)
            self.chargings_radiation.append(radiation_sprite)
        for u in self.charging_goal_infos:
            user_sprite = pyglet.sprite.Sprite(img=user, x=u['position_x'] - 7., y=u['position_y'] - 7.)
            self.charging_goals.append(user_sprite)


    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.charging_infos, self.charging_goal_infos, self.region_x, self.region_y)
        self.viewer.render()

    def sample_action(self, charging_num):
        return [np.random.uniform(-1, 1) for _ in range(charging_num * 2)]


class Viewer(pyglet.window.Window):
    def __init__(self, charging_infos, charging_goal_infos, x, y):
        self.charging_infos = charging_infos
        self.charging_goal_infos = charging_goal_infos
        pyglet.resource.path = ['resources']
        pyglet.resource.reindex()
        charging = pyglet.resource.image('charging.png')
        uav = pyglet.resource.image('uav.png')
        user = pyglet.resource.image('user.png')
        radiation = pyglet.resource.image('radiation.png')
        super(Viewer, self).__init__(width=x, height=y, resizable=False, caption='Multi-Agent', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.batch = pyglet.graphics.Batch()  # display whole batch at once
        self.chargings = []
        self.chargings_radiation = []
        self.charging_goals = []
        for u in self.charging_infos:
            charging_sprite = pyglet.sprite.Sprite(img=charging, x=u['position_x'] - 17., y=u['position_y'] - 10.)
            radiation_sprite = pyglet.sprite.Sprite(img=radiation, x=u['position_x'] - 100., y=u['position_y'] - 100.)
            uav_sprite = pyglet.sprite.Sprite(img=uav, x=u['position_x'] - 100., y=u['position_y'] - 100.)
            self.chargings.append(charging_sprite)
            self.chargings_radiation.append(radiation_sprite)
        for u in self.charging_goal_infos:
            user_sprite = pyglet.sprite.Sprite(img=user, x=u['position_x'] - 7., y=u['position_y'] - 7.)
            self.charging_goals.append(user_sprite)

    def render(self):
        self._update_charging()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        for charging in self.chargings:
            charging.draw()
        for charging_radiation in self.chargings_radiation:
            charging_radiation.draw()
        for charging_goal in self.charging_goals:
            charging_goal.draw()

    def _update_charging(self):

        for i in range(np.size(self.charging_infos)):
            self.chargings[i].update(x=(self.charging_infos[i])['position_x'] - 17., y=(self.charging_infos[i])['position_y'] - 17.)
            self.chargings_radiation[i].update(x=(self.charging_infos[i])['position_x'] - 100.,
                                          y=(self.charging_infos[i])['position_y'] - 100.)
        for i in range(np.size(self.charging_goal_infos)):
            self.charging_goals[i].update(x=(self.charging_goal_infos[i])['position_x'] - 7.,
                                     y=(self.charging_goal_infos[i])['position_y'] - 7.)


if __name__ == '__main__':
    env = ArmEnv(3, 20)
    while True:
        env.render()
        env.step(env.sample_action(3))