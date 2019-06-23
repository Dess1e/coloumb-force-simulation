from random import randrange
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

matplotlib.use('TkAgg')


def coloumb1(c, dist):
    return c ** 2 / dist ** 2

def coloumb2(c1, c2, pos1, pos2):
    diff = pos1 - pos2
    dist = np.linalg.norm(diff)
    unit = diff / dist  # normalized distance vector
    return (c1 * c2 * unit) / dist ** 2


class Particle:
    BORDER_COLLIDE_DAMPING_COEFFICIENT = 0.75
    COMMON_VELOCITY_DROP_COEFFICIENT = 1

    def __init__(self, x, y, velocity=None, mass=1, charge=1, name=None):
        self.pos = np.array([x, y], dtype=np.float)
        self._nextpos = None
        self.mass = mass
        self.charge = charge
        self.velocity = np.array([0.0, 0.0]) if velocity is None else np.array(velocity, dtype=np.float)
        self.accel = np.array([0.0, 0.0])

        if name is None:
            self._name = f'particle{randrange(1, 10 * 5)}'

    def __repr__(self):
        return self._name

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    def tick(self, others):
        for part in others:
            assert isinstance(part, Particle)
            force = coloumb2(self.charge, part.charge, self.pos, part.pos)
            self.accel = force / self.mass
            self.velocity += self.accel
            self.velocity *= self.COMMON_VELOCITY_DROP_COEFFICIENT
            self._nextpos = self.pos + self.velocity

    def move(self):
        assert self._nextpos is not None
        self.check_for_boundaries()
        self.pos = self._nextpos
        self._nextpos = None

    def check_for_boundaries(self):
        axes = plt.gca()
        x_min, x_max = axes.get_xlim()
        y_min, y_max = axes.get_ylim()
        x_basis = np.array([1.0, 0.0])
        y_basis = np.array([0.0, 1.0])
        x, y = self._nextpos
        if x <= x_min:
            nrm = x_basis
        elif x >= x_max:
            nrm = -x_basis
        elif y <= y_min:
            nrm = y_basis
        elif y >= y_max:
            nrm = -y_basis
        else:
            return

        reflected_velocity = self.velocity - 2 * np.dot(self.velocity, nrm) * nrm
        reflected_velocity *= self.BORDER_COLLIDE_DAMPING_COEFFICIENT
        self.velocity = reflected_velocity
        self._nextpos = self.pos + self.velocity


class ExcluderList(list):
    def without(self, elem):
        cpy = self.copy()
        cpy.remove(elem)
        return cpy


def main():
    fig, ax = plt.subplots()

    pos_line, = plt.plot([], [], 'b.')
    neg_line, = plt.plot([], [], 'r.')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.xticks([], [])
    plt.yticks([], [])
    particles = ExcluderList([
        Particle(90, 90),
        Particle(30, 10),
        Particle(30, 20, charge=-1),
        Particle(20, 20, charge=-1),
    ])

    def update(frame):
        [p.tick(particles.without(p)) for p in particles]
        [p.move() for p in particles]

        pos = list(filter(
            lambda p: p.charge > 0,
            particles
        ))
        neg = list(filter(
            lambda p: p.charge < 0,
            particles
        ))

        neg_line.set_data([p.x for p in neg], [p.y for p in neg])
        pos_line.set_data([p.x for p in pos], [p.y for p in pos])

        return neg_line, pos_line

    anim = FuncAnimation(
        fig,
        update,
        frames=2000,
        interval=5,
        blit=True
    )

    anim.save('rep_and_attr.mp4', fps=120)
    # plt.show()


if __name__ == '__main__':
    main()
